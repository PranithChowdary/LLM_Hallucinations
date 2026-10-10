"""
Generate the A1/A4 data report (Markdown) from the unified schema.

Examples:

    python -m src.data.report --source hf --out data_report.md

    python -m src.data.report \
        --source local \
        --halueval-dir path/to/HaluEval/data \
        --truthfulqa-csv TruthfulQA.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data.loaders import load_all, to_candidates


def wc(value) -> int:
    """Return whitespace-separated word count."""
    if value is None:
        return 0

    if pd.isna(value):
        return 0

    return len(str(value).split())


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute AUROC using Mann-Whitney U with tie-averaged ranks.

    Returns:
        0.5 -> no discrimination
        1.0 -> perfect ranking
        0.0 -> perfectly reversed ranking
        NaN -> only one class is present
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")

    unique_labels = np.unique(labels)

    if not set(unique_labels).issubset({0, 1}):
        raise ValueError("labels must contain only 0 and 1")

    n1 = int((labels == 1).sum())
    n0 = int((labels == 0).sum())

    if n1 == 0 or n0 == 0:
        return float("nan")

    ranks = pd.Series(scores).rank(method="average").to_numpy()

    positive_rank_sum = ranks[labels == 1].sum()

    return float(
        (
            positive_rank_sum
            - n1 * (n1 + 1) / 2
        )
        / (n1 * n0)
    )


def md_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame into a simple Markdown table."""
    df = df.reset_index()

    header = (
        "| "
        + " | ".join(map(str, df.columns))
        + " |\n"
        + "|"
        + "---|" * len(df.columns)
        + "\n"
    )

    rows = []

    for row in df.itertuples(index=False):
        formatted = []

        for value in row:
            if isinstance(value, (float, np.floating)):
                if np.isnan(value):
                    formatted.append("NaN")
                else:
                    formatted.append(f"{value:.2f}")
            else:
                formatted.append(str(value))

        rows.append("| " + " | ".join(formatted) + " |")

    body = "\n".join(rows)

    return header + body + "\n"


def quantiles(series: pd.Series) -> dict:
    """Return basic length statistics."""
    series = series.dropna()

    if len(series) == 0:
        return {
            "mean": np.nan,
            "p5": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    q = series.quantile([0.05, 0.50, 0.95])

    return {
        "mean": series.mean(),
        "p5": q.loc[0.05],
        "median": q.loc[0.50],
        "p95": q.loc[0.95],
        "max": series.max(),
    }


def build_report(df: pd.DataFrame, seed: int = 0) -> str:
    """
    Build the A1/A4 Markdown data report.

    The report covers:
        1. Dataset size and reference-label balance
        2. Prompt/context/response length distributions
        3. Potential data-quality and leakage issues
        4. Cross-subset prompt overlap
        5. TruthfulQA category distribution
        6. Small random sample for manual inspection
    """

    out = [
        "# Data report: HaluEval + TruthfulQA\n"
    ]

    # ---------------------------------------------------------
    # Convert reference lists into one row per candidate response
    # ---------------------------------------------------------

    cand = to_candidates(df)

    cand["n_words"] = cand["response"].map(wc)

    # =========================================================
    # 1. SIZE AND LABEL BALANCE
    # =========================================================

    out.append(
        "## 1. Size and label balance (reference answers)\n"
    )

    sizes = (
        df.groupby(["dataset", "subset"])
        .agg(prompts=("uid", "count"))
    )

    balance = (
        cand.groupby(["dataset", "subset", "label"])
        .size()
        .unstack(fill_value=0)
        .rename(
            columns={
                0: "faithful_refs",
                1: "hallucinated_refs",
            }
        )
    )

    # Make sure both columns exist even if one label is absent.
    if "faithful_refs" not in balance.columns:
        balance["faithful_refs"] = 0

    if "hallucinated_refs" not in balance.columns:
        balance["hallucinated_refs"] = 0

    balance["frac_hallucinated"] = (
        balance["hallucinated_refs"]
        / (
            balance["faithful_refs"]
            + balance["hallucinated_refs"]
        )
    )

    sizes = sizes.join(balance)

    out.append(md_table(sizes))

    out.append(
        "HaluEval is balanced at the reference-answer level "
        "by construction: each prompt provides one correct/right "
        "reference and one hallucinated reference. TruthfulQA has "
        "variable numbers of correct and incorrect reference answers "
        "per question. Therefore, this table describes the reference "
        "data, not the eventual label balance of our generated answers. "
        "The actual generation-level label balance is determined later "
        "in A2 when model generations are evaluated.\n"
    )

    # =========================================================
    # 2. LENGTH DISTRIBUTIONS
    # =========================================================

    out.append(
        "## 2. Length distributions (words)\n"
    )

    rows = []

    for subset, group in df.groupby("subset"):

        # Prompt length
        rows.append(
            {
                "subset": subset,
                "field": "prompt",
                **quantiles(group["prompt"].map(wc)),
            }
        )

        # Context length
        context = group["context"].dropna()

        if len(context) > 0:
            rows.append(
                {
                    "subset": subset,
                    "field": "context",
                    **quantiles(context.map(wc)),
                }
            )

        # Reference response lengths
        subset_candidates = cand[
            cand["subset"] == subset
        ]

        for label, name in [
            (0, "faithful response"),
            (1, "hallucinated response"),
        ]:
            response_lengths = subset_candidates[
                subset_candidates["label"] == label
            ]["n_words"]

            rows.append(
                {
                    "subset": subset,
                    "field": name,
                    **quantiles(response_lengths),
                }
            )

    length_df = pd.DataFrame(rows)

    out.append(
        md_table(
            length_df.set_index(["subset", "field"])
        )
    )

    # =========================================================
    # 3. POTENTIAL PROBLEMS / ODDITIES
    # =========================================================

    out.append(
        "## 3. Things that might look off\n"
    )

    checks = []

    for subset, group in df.groupby("subset"):

        subset_candidates = cand[
            cand["subset"] == subset
        ]

        # Empty prompts
        empty_prompts = int(
            group["prompt"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        # Empty reference responses
        empty_responses = int(
            subset_candidates["response"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        n_empty = empty_prompts + empty_responses

        # Duplicate normalized prompts
        duplicate_prompt_rows = int(
            group.duplicated(
                "norm_key",
                keep=False
            ).sum()
        )

        # Check whether the same exact reference appears
        # in both correct and incorrect lists.
        identical_refs = 0

        # Crude substring-overlap diagnostic.
        substring_overlaps = 0

        for row in group.itertuples():

            correct = {
                str(x).strip().lower()
                for x in row.reference_correct
                if str(x).strip()
            }

            incorrect = {
                str(x).strip().lower()
                for x in row.reference_incorrect
                if str(x).strip()
            }

            identical_refs += len(
                correct.intersection(incorrect)
            )

            # Skip this heuristic for summarization because
            # summary/reference text naturally overlaps heavily.
            if subset != "summarization":

                for bad in incorrect:
                    for good in correct:

                        if (
                            bad
                            and good
                            and bad != good
                            and (
                                bad in good
                                or good in bad
                            )
                        ):
                            substring_overlaps += 1

        # How informative is response length alone?
        length_auc = auroc(
            subset_candidates["n_words"].to_numpy(),
            subset_candidates["label"].to_numpy(),
        )

        checks.append(
            {
                "subset": subset,
                "empty_fields": n_empty,
                "rows_sharing_a_duplicate_prompt": (
                    duplicate_prompt_rows
                ),
                "identical_right_and_hallucinated": (
                    identical_refs
                ),
                "hallucinated_overlaps_right_substring": (
                    substring_overlaps
                ),
                "length_only_AUROC": length_auc,
            }
        )

    checks_df = (
        pd.DataFrame(checks)
        .set_index("subset")
    )

    out.append(md_table(checks_df))

    out.append(
        "`length_only_AUROC` measures how well response length "
        "alone predicts the reference hallucinated label. A value "
        "near 0.50 means length provides little discrimination, while "
        "a value farther from 0.50 indicates a possible length shortcut "
        "in the reference data. This matters because output length is "
        "also one of our Tier-1 single-pass features. Any substantial "
        "shortcut should therefore be discussed in the paper.\n"
    )

    out.append(
        "The substring-overlap column is only a diagnostic heuristic. "
        "It should not be interpreted as proof that a reference pair "
        "is duplicated or incorrectly labeled.\n"
    )

    # =========================================================
    # 4. CROSS-SUBSET OVERLAP
    # =========================================================

    out.append(
        "## 4. Cross-subset normalized-prompt overlap\n"
    )

    keys = {
        subset: set(group["norm_key"])
        for subset, group in df.groupby("subset")
    }

    overlap_rows = []

    subset_names = sorted(keys)

    for i, first in enumerate(subset_names):
        for second in subset_names[i + 1:]:

            overlap_rows.append(
                {
                    "a": first,
                    "b": second,
                    "shared_normalized_prompts": len(
                        keys[first].intersection(
                            keys[second]
                        )
                    ),
                }
            )

    if overlap_rows:
        overlap_df = pd.DataFrame(
            overlap_rows
        ).set_index(["a", "b"])

        out.append(md_table(overlap_df))

    else:
        out.append(
            "No cross-subset comparisons available.\n"
        )

    out.append(
        "Shared normalized prompts are potential leakage risks for "
        "the calibration/test split. They should be investigated before "
        "splitting. This check is intentionally conservative because "
        "`norm_key` is based on normalized prompt text rather than "
        "semantic similarity.\n"
    )

    # =========================================================
    # 5. TRUTHFULQA CATEGORY DISTRIBUTION
    # =========================================================

    truthfulqa = df[
        df["subset"] == "truthfulqa"
    ]

    if len(truthfulqa) > 0:

        out.append(
            "## 5. TruthfulQA questions per category (top 10)\n"
        )

        category_counts = (
            truthfulqa["category"]
            .fillna("Unknown")
            .value_counts()
            .head(10)
            .rename("n")
            .to_frame()
        )

        out.append(
            md_table(category_counts)
        )

    # =========================================================
    # 6. MANUAL READING SAMPLE
    # =========================================================

    out.append(
        "## 6. Random sample to read by hand\n"
    )

    rng = np.random.default_rng(seed)

    for subset, group in df.groupby("subset"):

        n = min(3, len(group))

        if n == 0:
            continue

        selected_indices = rng.choice(
            len(group),
            size=n,
            replace=False,
        )

        for index in selected_indices:

            row = group.iloc[index]

            correct = (
                row["reference_correct"][0]
                if row["reference_correct"]
                else ""
            )

            incorrect = (
                row["reference_incorrect"][0]
                if row["reference_incorrect"]
                else ""
            )

            out.append(
                f"**{row['uid']}**\n"
                f"- prompt: {str(row['prompt'])[:300]}\n"
                f"- faithful: {str(correct)[:300]}\n"
                f"- hallucinated: {str(incorrect)[:300]}\n"
            )

    return "\n".join(out)


def main() -> None:
    """Command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a Markdown data report from "
            "the unified HaluEval + TruthfulQA schema."
        )
    )

    parser.add_argument(
        "--source",
        default="hf",
        choices=["hf", "local"],
        help="Data source.",
    )

    parser.add_argument(
        "--halueval-dir",
        default=None,
        help="Directory containing local HaluEval JSON files.",
    )

    parser.add_argument(
        "--truthfulqa-csv",
        default=None,
        help="Path to local TruthfulQA CSV file.",
    )

    parser.add_argument(
        "--out",
        default="data_report.md",
        help="Output Markdown file.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the manual-reading sample.",
    )

    args = parser.parse_args()

    frame = load_all(
        source=args.source,
        halueval_dir=args.halueval_dir,
        truthfulqa_csv=args.truthfulqa_csv,
    )

    report = build_report(
        frame,
        seed=args.seed,
    )

    with open(
        args.out,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    print(
        f"wrote {args.out} "
        f"({len(frame)} prompts)"
    )


if __name__ == "__main__":
    main()