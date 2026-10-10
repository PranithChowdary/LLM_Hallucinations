"""
A4 - Generation Sanity Audit

Audits the generated calibration set for:
1. Generation count and UID integrity
2. Dataset/subset distribution
3. Generated-answer length statistics
4. Empty / very short / very long answers
5. Repeated and degenerate outputs
6. Refusal-like outputs
7. Generation metadata consistency
8. UID integrity against the frozen calibration set

Input:
    data/generation_qwen_calibration.parquet

Output:
    data/generation_audit.md
"""

from pathlib import Path
import re

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GENERATION_FILE = (
    PROJECT_ROOT / "data" / "generation_qwen_calibration.parquet"
)

# Change this only if your frozen calibration file has a different name.
CALIBRATION_CANDIDATES = [
    PROJECT_ROOT / "data" / "calibration.parquet",
    PROJECT_ROOT / "data" / "frozen_calibration.parquet",
    PROJECT_ROOT / "data" / "calibration_set.parquet",
]

OUTPUT_FILE = PROJECT_ROOT / "data" / "generation_audit.md"

EXPECTED_COUNT = 24652

# Thresholds for sanity checks
VERY_SHORT_WORDS = 3
VERY_LONG_WORDS = 500

# Used to identify obvious refusal-style answers.
REFUSAL_PATTERNS = [
    r"\bi can't\b",
    r"\bi cannot\b",
    r"\bi'm unable\b",
    r"\bi am unable\b",
    r"\bi don't know\b",
    r"\bi do not know\b",
    r"\bi cannot answer\b",
    r"\bi can't answer\b",
    r"\bi'm not able\b",
    r"\bi am not able\b",
    r"\bas an ai\b",
]


# ============================================================
# Helper functions
# ============================================================

def word_count(text: str) -> int:
    """Count whitespace-separated words."""
    if not isinstance(text, str):
        return 0
    return len(text.split())


def char_count(text: str) -> int:
    """Count characters."""
    if not isinstance(text, str):
        return 0
    return len(text)


def refusal_like(text: str) -> bool:
    """Return True if text contains an obvious refusal pattern."""
    if not isinstance(text, str):
        return False

    text_lower = text.lower()

    return any(
        re.search(pattern, text_lower)
        for pattern in REFUSAL_PATTERNS
    )


def repeated_word_ratio(text: str) -> float:
    """
    Calculate the proportion of words belonging to the most
    frequently repeated word.

    This is only an indicator of obvious repetition.
    """
    if not isinstance(text, str):
        return 0.0

    words = re.findall(r"\b\w+\b", text.lower())

    if not words:
        return 0.0

    counts = pd.Series(words).value_counts()

    return float(counts.iloc[0] / len(words))


def repeated_ngram(text: str, n: int = 3):
    """
    Find the most repeated n-gram.

    Returns:
        (ngram, count)
    """
    if not isinstance(text, str):
        return None, 0

    words = re.findall(r"\b\w+\b", text.lower())

    if len(words) < n:
        return None, 0

    ngrams = [
        " ".join(words[i:i + n])
        for i in range(len(words) - n + 1)
    ]

    counts = pd.Series(ngrams).value_counts()

    if len(counts) == 0:
        return None, 0

    return counts.index[0], int(counts.iloc[0])


def find_calibration_file():
    """Find a likely frozen calibration file."""
    for path in CALIBRATION_CANDIDATES:
        if path.exists():
            return path

    return None


def format_percentage(value: float) -> str:
    return f"{value:.2f}%"


# ============================================================
# Main audit
# ============================================================

def main():

    print("=" * 70)
    print("A4 - GENERATION SANITY AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load generation data
    # --------------------------------------------------------

    if not GENERATION_FILE.exists():
        raise FileNotFoundError(
            f"Generation file not found:\n{GENERATION_FILE}"
        )

    df = pd.read_parquet(GENERATION_FILE)

    print(f"\nGeneration file: {GENERATION_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    required_columns = [
        "uid",
        "dataset",
        "subset",
        "prompt",
        "generated_answer",
        "model",
        "seed",
        "temperature",
        "top_p",
        "max_new_tokens",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # 2. Basic count
    # --------------------------------------------------------

    actual_count = len(df)
    count_ok = actual_count == EXPECTED_COUNT

    # --------------------------------------------------------
    # 3. UID checks
    # --------------------------------------------------------

    uid_null_count = int(df["uid"].isna().sum())
    duplicate_uid_count = int(df["uid"].duplicated().sum())
    unique_uid_count = int(df["uid"].nunique())

    # --------------------------------------------------------
    # 4. Dataset/subset distribution
    # --------------------------------------------------------

    distribution = (
        df.groupby(["dataset", "subset"])
        .size()
        .reset_index(name="count")
        .sort_values(["dataset", "subset"])
    )

    # --------------------------------------------------------
    # 5. Generated-answer lengths
    # --------------------------------------------------------

    df["answer_words"] = df["generated_answer"].apply(word_count)
    df["answer_chars"] = df["generated_answer"].apply(char_count)

    lengths = df["answer_words"]

    length_stats = {
        "mean": lengths.mean(),
        "median": lengths.median(),
        "min": lengths.min(),
        "max": lengths.max(),
        "p5": lengths.quantile(0.05),
        "p95": lengths.quantile(0.95),
    }

    # --------------------------------------------------------
    # 6. Empty / short / long answers
    # --------------------------------------------------------

    empty_mask = (
        df["generated_answer"].isna()
        | df["generated_answer"].astype(str).str.strip().eq("")
    )

    very_short_mask = (
        (~empty_mask)
        & (df["answer_words"] <= VERY_SHORT_WORDS)
    )

    very_long_mask = (
        df["answer_words"] > VERY_LONG_WORDS
    )

    empty_count = int(empty_mask.sum())
    very_short_count = int(very_short_mask.sum())
    very_long_count = int(very_long_mask.sum())

    # --------------------------------------------------------
    # 7. Exact duplicate generated answers
    # --------------------------------------------------------

    normalized_answers = (
        df["generated_answer"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    duplicate_answer_count = int(
        normalized_answers.duplicated(keep=False).sum()
    )

    unique_answer_count = int(normalized_answers.nunique())

    # Answers that occur more than once
    answer_frequency = normalized_answers.value_counts()

    repeated_answer_groups = int(
        (answer_frequency > 1).sum()
    )

    max_same_answer_count = (
        int(answer_frequency.iloc[0])
        if len(answer_frequency) > 0
        else 0
    )

    # --------------------------------------------------------
    # 8. Repeated-word / degenerate outputs
    # --------------------------------------------------------

    df["repeated_word_ratio"] = (
        df["generated_answer"]
        .apply(repeated_word_ratio)
    )

    # Flag if one word accounts for >= 20% of all words.
    high_word_repetition_mask = (
        df["answer_words"] >= 10
    ) & (
        df["repeated_word_ratio"] >= 0.20
    )

    high_word_repetition_count = int(
        high_word_repetition_mask.sum()
    )

    # --------------------------------------------------------
    # 9. Repeated trigram detection
    # --------------------------------------------------------

    repeated_ngram_counts = []

    for answer in df["generated_answer"]:
        _, count = repeated_ngram(answer, n=3)
        repeated_ngram_counts.append(count)

    df["max_repeated_trigram_count"] = repeated_ngram_counts

    # Flag obvious repeated 3-gram loops.
    repeated_trigram_mask = (
        df["max_repeated_trigram_count"] >= 5
    )

    repeated_trigram_count = int(
        repeated_trigram_mask.sum()
    )

    # --------------------------------------------------------
    # 10. Refusal-like answers
    # --------------------------------------------------------

    df["refusal_like"] = (
        df["generated_answer"]
        .apply(refusal_like)
    )

    refusal_count = int(df["refusal_like"].sum())

    # --------------------------------------------------------
    # 11. Metadata checks
    # --------------------------------------------------------

    metadata_columns = [
        "model",
        "seed",
        "temperature",
        "top_p",
        "max_new_tokens",
    ]

    metadata_summary = {}

    for column in metadata_columns:
        values = df[column].drop_duplicates().tolist()
        metadata_summary[column] = values

    # --------------------------------------------------------
    # 12. UID format checks
    # --------------------------------------------------------

    uid_pattern = re.compile(
    r"^(halueval-(qa|dialogue|summarization)-\d+"
    r"|truthfulqa-\d+)$"
    )

    invalid_uid_mask = ~df["uid"].astype(str).apply(
        lambda x: bool(uid_pattern.match(x))
    )

    invalid_uid_count = int(invalid_uid_mask.sum())

    # --------------------------------------------------------
    # 13. Compare against frozen calibration set
    # --------------------------------------------------------

    calibration_file = find_calibration_file()

    calibration_check = "Not performed — calibration file not found."

    missing_from_generation = []
    extra_in_generation = []

    if calibration_file is not None:

        calibration_df = pd.read_parquet(calibration_file)

        if "uid" not in calibration_df.columns:
            calibration_check = (
                f"Calibration file found at `{calibration_file}`, "
                "but it has no `uid` column."
            )
        else:

            calibration_uids = set(
                calibration_df["uid"].dropna().astype(str)
            )

            generation_uids = set(
                df["uid"].dropna().astype(str)
            )

            missing_from_generation = sorted(
                calibration_uids - generation_uids
            )

            extra_in_generation = sorted(
                generation_uids - calibration_uids
            )

            calibration_check = (
                f"Compared against `{calibration_file}`."
            )

    # --------------------------------------------------------
    # 14. Build Markdown report
    # --------------------------------------------------------

    report = []

    report.append("# A4 — Generation Sanity Audit")
    report.append("")
    report.append(
        "Automated sanity audit of "
        "`generation_qwen_calibration.parquet`."
    )
    report.append("")

    # --------------------------------------------------------
    # Overview
    # --------------------------------------------------------

    report.append("## 1. Overview")
    report.append("")
    report.append("| Check | Result |")
    report.append("|---|---:|")
    report.append(
        f"| Expected generations | {EXPECTED_COUNT:,} |"
    )
    report.append(
        f"| Actual generations | {actual_count:,} |"
    )
    report.append(
        f"| Count matches expected | {'PASS' if count_ok else 'FAIL'} |"
    )
    report.append(
        f"| Unique UIDs | {unique_uid_count:,} |"
    )
    report.append(
        f"| Duplicate UIDs | {duplicate_uid_count:,} |"
    )
    report.append(
        f"| Null UIDs | {uid_null_count:,} |"
    )
    report.append(
        f"| Invalid UID format | {invalid_uid_count:,} |"
    )
    report.append("")

    # --------------------------------------------------------
    # Dataset distribution
    # --------------------------------------------------------

    report.append("## 2. Dataset / Subset Distribution")
    report.append("")
    report.append("| Dataset | Subset | Count | Percentage |")
    report.append("|---|---|---:|---:|")

    for _, row in distribution.iterrows():

        percentage = (
            row["count"] / actual_count * 100
            if actual_count
            else 0
        )

        report.append(
            f"| {row['dataset']} | {row['subset']} | "
            f"{int(row['count']):,} | "
            f"{percentage:.2f}% |"
        )

    report.append("")

    # --------------------------------------------------------
    # Length distribution
    # --------------------------------------------------------

    report.append("## 3. Generated-Answer Length Distribution")
    report.append("")
    report.append("| Statistic | Words |")
    report.append("|---|---:|")

    report.append(
        f"| Mean | {length_stats['mean']:.2f} |"
    )
    report.append(
        f"| Median | {length_stats['median']:.2f} |"
    )
    report.append(
        f"| Minimum | {int(length_stats['min'])} |"
    )
    report.append(
        f"| Maximum | {int(length_stats['max'])} |"
    )
    report.append(
        f"| P5 | {length_stats['p5']:.2f} |"
    )
    report.append(
        f"| P95 | {length_stats['p95']:.2f} |"
    )

    report.append("")

    # --------------------------------------------------------
    # Degenerate outputs
    # --------------------------------------------------------

    report.append("## 4. Degenerate / Suspicious Outputs")
    report.append("")
    report.append("| Check | Count | Percentage |")
    report.append("|---|---:|---:|")

    report.append(
        f"| Empty answers | {empty_count:,} | "
        f"{format_percentage(empty_count / actual_count * 100)} |"
    )

    report.append(
        f"| Very short (≤ {VERY_SHORT_WORDS} words) | "
        f"{very_short_count:,} | "
        f"{format_percentage(very_short_count / actual_count * 100)} |"
    )

    report.append(
        f"| Very long (> {VERY_LONG_WORDS} words) | "
        f"{very_long_count:,} | "
        f"{format_percentage(very_long_count / actual_count * 100)} |"
    )

    report.append(
        f"| Exact duplicate answers | "
        f"{duplicate_answer_count:,} | "
        f"{format_percentage(duplicate_answer_count / actual_count * 100)} |"
    )

    report.append(
        f"| Answer strings occurring >1 time | "
        f"{repeated_answer_groups:,} |"
    )

    report.append(
        f"| Maximum copies of same answer | "
        f"{max_same_answer_count:,} |"
    )

    report.append(
        f"| High single-word repetition | "
        f"{high_word_repetition_count:,} | "
        f"{format_percentage(high_word_repetition_count / actual_count * 100)} |"
    )

    report.append(
        f"| Repeated trigram ≥5 times | "
        f"{repeated_trigram_count:,} | "
        f"{format_percentage(repeated_trigram_count / actual_count * 100)} |"
    )

    report.append(
        f"| Refusal-like answers | "
        f"{refusal_count:,} | "
        f"{format_percentage(refusal_count / actual_count * 100)} |"
    )

    report.append("")

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    report.append("## 5. Generation Metadata")
    report.append("")

    for column, values in metadata_summary.items():

        report.append(f"### `{column}`")
        report.append("")

        for value in values:
            report.append(f"- `{value}`")

        report.append("")

    # --------------------------------------------------------
    # UID integrity
    # --------------------------------------------------------

    report.append("## 6. UID Integrity")
    report.append("")

    if calibration_file is None:

        report.append(
            "⚠️ Frozen calibration file was not automatically found."
        )
        report.append("")
        report.append(calibration_check)

    else:

        report.append(
            f"Calibration file: `{calibration_file}`"
        )
        report.append("")

        report.append(
            f"- Missing from generation: "
            f"**{len(missing_from_generation):,}**"
        )

        report.append(
            f"- Extra in generation: "
            f"**{len(extra_in_generation):,}**"
        )

        if not missing_from_generation and not extra_in_generation:

            report.append("")
            report.append(
                "✅ Generation UIDs exactly match the frozen "
                "calibration UID set."
            )

        else:

            report.append("")

            if missing_from_generation:

                report.append("### Missing UIDs")
                report.append("")

                for uid in missing_from_generation[:50]:
                    report.append(f"- `{uid}`")

                if len(missing_from_generation) > 50:
                    report.append(
                        f"- ... and "
                        f"{len(missing_from_generation) - 50:,} more"
                    )

            if extra_in_generation:

                report.append("### Extra UIDs")
                report.append("")

                for uid in extra_in_generation[:50]:
                    report.append(f"- `{uid}`")

                if len(extra_in_generation) > 50:
                    report.append(
                        f"- ... and "
                        f"{len(extra_in_generation) - 50:,} more"
                    )

    report.append("")

    # --------------------------------------------------------
    # Examples of suspicious outputs
    # --------------------------------------------------------

    report.append("## 7. Examples of Suspicious Outputs")
    report.append("")

    suspicious_mask = (
        empty_mask
        | very_short_mask
        | very_long_mask
        | high_word_repetition_mask
        | repeated_trigram_mask
        | df["refusal_like"]
    )

    suspicious = df[suspicious_mask]

    if len(suspicious) == 0:

        report.append(
            "No outputs were flagged by the automated heuristics."
        )

    else:

        report.append(
            f"Total flagged by at least one heuristic: "
            f"**{len(suspicious):,}**"
        )
        report.append("")

        report.append(
            "> These are heuristic flags, not automatic hallucination labels."
        )
        report.append("")

        for _, row in suspicious.head(20).iterrows():

            answer = str(row["generated_answer"])

            if len(answer) > 1000:
                answer = answer[:1000] + "..."

            answer = answer.replace("\n", " ")

            report.append(
                f"### `{row['uid']}`"
            )
            report.append("")
            report.append(
                f"**Dataset:** `{row['dataset']}`  "
                f"**Subset:** `{row['subset']}`"
            )
            report.append("")
            report.append(
                f"**Length:** {row['answer_words']} words"
            )
            report.append("")
            report.append(
                f"**Answer:** {answer}"
            )
            report.append("")

    # --------------------------------------------------------
    # Overall interpretation
    # --------------------------------------------------------

    report.append("## 8. Overall A4 Interpretation")
    report.append("")

    major_failures = []

    if not count_ok:
        major_failures.append(
            "generation count does not match 24,652"
        )

    if uid_null_count > 0:
        major_failures.append("null UIDs exist")

    if duplicate_uid_count > 0:
        major_failures.append("duplicate UIDs exist")

    if invalid_uid_count > 0:
        major_failures.append("invalid UID formats exist")

    if calibration_file is not None:
        if missing_from_generation:
            major_failures.append(
                "some frozen calibration UIDs are missing"
            )

        if extra_in_generation:
            major_failures.append(
                "generation contains UIDs outside the frozen calibration set"
            )

    if major_failures:

        report.append("### ⚠️ Review required")
        report.append("")

        for failure in major_failures:
            report.append(f"- {failure}")

        report.append("")
        report.append(
            "Do not proceed to A3 until these structural issues "
            "are understood and resolved."
        )

    else:

        report.append("### ✅ Structural checks passed")
        report.append("")
        report.append(
            "The generation set passed the major structural integrity "
            "checks. Suspicious-output statistics should still be "
            "reviewed before proceeding to A3."
        )

    report.append("")
    report.append(
        "**Important:** A4 is a generation sanity audit. "
        "It does not determine whether an individual answer is "
        "actually hallucinated."
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(f"\nReport written to:")
    print(OUTPUT_FILE)

    print("\nKey results:")
    print(f"  Expected rows:       {EXPECTED_COUNT:,}")
    print(f"  Actual rows:         {actual_count:,}")
    print(f"  Unique UIDs:         {unique_uid_count:,}")
    print(f"  Duplicate UIDs:      {duplicate_uid_count:,}")
    print(f"  Empty answers:       {empty_count:,}")
    print(f"  Very short answers:  {very_short_count:,}")
    print(f"  Very long answers:   {very_long_count:,}")
    print(f"  Duplicate answers:   {duplicate_answer_count:,}")
    print(f"  Refusal-like:        {refusal_count:,}")

    if calibration_file is not None:
        print(
            f"  Missing calibration UIDs: "
            f"{len(missing_from_generation):,}"
        )
        print(
            f"  Extra generation UIDs:    "
            f"{len(extra_in_generation):,}"
        )


if __name__ == "__main__":
    main()