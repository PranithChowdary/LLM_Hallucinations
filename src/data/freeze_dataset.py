from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.data.loaders import load_all


RANDOM_SEED = 42
N_SPLITS = 5

DATA_DIR = Path("data")


def add_reference_conflict_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect answers that appear in both the correct and incorrect
    reference-answer lists.

    These rows are flagged, not deleted.
    """

    df = df.copy()

    conflict_flags = []
    conflict_answers = []

    for _, row in df.iterrows():

        correct = {
            str(x).strip().lower()
            for x in row["reference_correct"]
            if str(x).strip()
        }

        incorrect = {
            str(x).strip().lower()
            for x in row["reference_incorrect"]
            if str(x).strip()
        }

        overlap = sorted(correct.intersection(incorrect))

        if overlap:
            conflict_flags.append(True)
            conflict_answers.append(overlap)
        else:
            conflict_flags.append(False)
            conflict_answers.append([])

    df["reference_conflict"] = conflict_flags
    df["conflict_answers"] = conflict_answers

    return df


def remove_exact_duplicate_groups(df: pd.DataFrame):
    """
    Remove exact duplicate normalized prompt/source groups.

    One deterministic representative is kept from each group.
    """

    df = df.copy()

    duplicate_mask = df.duplicated(
        subset=["norm_key"],
        keep=False
    )

    duplicate_rows = df[duplicate_mask].copy()

    if duplicate_rows.empty:
        return df, duplicate_rows, pd.DataFrame()

    duplicate_rows = duplicate_rows.sort_values(
        ["norm_key", "uid"]
    )

    keep_mask = ~df.duplicated(
        subset=["norm_key"],
        keep="first"
    )

    cleaned = df[keep_mask].copy()

    removed = df[~keep_mask].copy()

    duplicate_groups = (
        duplicate_rows
        .groupby("norm_key", as_index=False)
        .agg(
            duplicate_count=("uid", "count"),
            uids=("uid", list),
        )
    )

    return cleaned, removed, duplicate_groups


def make_stratified_group_split(df: pd.DataFrame):
    """
    Create an approximately 80/20 calibration/test split.

    Stratification:
        dataset + subset

    Grouping:
        norm_key

    This prevents identical normalized prompts/sources
    from appearing in both calibration and test.
    """

    df = df.copy().reset_index(drop=True)

    df["split_stratum"] = (
        df["dataset"].astype(str)
        + "::"
        + df["subset"].astype(str)
    )

    groups = df["norm_key"].astype(str)

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            y=df["split_stratum"],
            groups=groups,
        )
    )

    calibration = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    calibration = calibration.drop(
        columns=["split_stratum"]
    )

    test = test.drop(
        columns=["split_stratum"]
    )

    return calibration, test


def verify_split(
    calibration: pd.DataFrame,
    test: pd.DataFrame
):
    """
    Verify that calibration and test have no leakage.
    """

    calibration_keys = set(
        calibration["norm_key"]
    )

    test_keys = set(
        test["norm_key"]
    )

    overlap = calibration_keys.intersection(
        test_keys
    )

    if overlap:
        raise RuntimeError(
            f"DATA LEAKAGE DETECTED: "
            f"{len(overlap)} norm_key values occur "
            f"in both splits."
        )

    uid_overlap = (
        set(calibration["uid"])
        .intersection(set(test["uid"]))
    )

    if uid_overlap:
        raise RuntimeError(
            "DATA LEAKAGE DETECTED: "
            "UID overlap between splits."
        )

    if len(calibration) + len(test) != 30815:
        raise RuntimeError(
            "Unexpected total split size."
        )

    print("Split leakage check: PASSED")
    print(f"Calibration rows: {len(calibration):,}")
    print(f"Test rows:        {len(test):,}")
    print(
        f"Total rows:       "
        f"{len(calibration) + len(test):,}"
    )


def print_split_summary(
    name: str,
    df: pd.DataFrame
):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(f"Rows: {len(df):,}")

    print("\nDataset/subset distribution:")

    summary = (
        df.groupby(
            ["dataset", "subset"]
        )
        .size()
        .reset_index(name="rows")
    )

    summary["percentage"] = (
        summary["rows"] / len(df) * 100
    ).round(2)

    print(
        summary.to_string(index=False)
    )

    print("\nReference-conflict rows:")

    conflict_count = int(
        df["reference_conflict"].sum()
    )

    print(
        f"{conflict_count} / {len(df)} "
        f"({conflict_count / len(df) * 100:.3f}%)"
    )


def main():

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("FREEZING CLEAN UNIFIED DATASET")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load original HaluEval + TruthfulQA data
    # ---------------------------------------------------------

    print(
        "\nLoading HaluEval + TruthfulQA..."
    )

    df = load_all(
        source="hf"
    )

    print(
        f"Original rows: {len(df):,}"
    )

    # ---------------------------------------------------------
    # 2. Detect reference conflicts
    # ---------------------------------------------------------

    print(
        "\nChecking reference conflicts..."
    )

    df = add_reference_conflict_flag(df)

    conflict_count = int(
        df["reference_conflict"].sum()
    )

    print(
        f"Reference-conflict rows detected: "
        f"{conflict_count}"
    )

    # ---------------------------------------------------------
    # 3. Remove exact duplicate groups
    # ---------------------------------------------------------

    print(
        "\nChecking exact duplicate groups..."
    )

    cleaned, removed, duplicate_groups = (
        remove_exact_duplicate_groups(df)
    )

    print(
        f"Rows removed as exact duplicates: "
        f"{len(removed):,}"
    )

    print(
        f"Cleaned dataset rows: "
        f"{len(cleaned):,}"
    )

    # ---------------------------------------------------------
    # 4. Save duplicate audit information
    # ---------------------------------------------------------

    if not removed.empty:

        removed_path = (
            DATA_DIR
            / "removed_duplicate_rows.parquet"
        )

        removed.to_parquet(
            removed_path,
            index=False
        )

        print(
            f"Duplicate audit saved to: "
            f"{removed_path}"
        )

    if not duplicate_groups.empty:

        duplicate_groups_path = (
            DATA_DIR
            / "duplicate_groups.csv"
        )

        duplicate_groups.to_csv(
            duplicate_groups_path,
            index=False
        )

        print(
            f"Duplicate groups saved to: "
            f"{duplicate_groups_path}"
        )

    # ---------------------------------------------------------
    # 5. Freeze unified dataset
    # ---------------------------------------------------------

    unified_path = (
        DATA_DIR
        / "unified_dataset.parquet"
    )

    cleaned.to_parquet(
        unified_path,
        index=False
    )

    print(
        f"\nUnified dataset saved to: "
        f"{unified_path}"
    )

    # ---------------------------------------------------------
    # 6. Create calibration/test split
    # ---------------------------------------------------------

    print(
        "\nCreating leakage-safe "
        "calibration/test split..."
    )

    calibration, test = (
        make_stratified_group_split(
            cleaned
        )
    )

    # ---------------------------------------------------------
    # 7. Verify split
    # ---------------------------------------------------------

    verify_split(
        calibration,
        test
    )

    # ---------------------------------------------------------
    # 8. Save splits
    # ---------------------------------------------------------

    calibration_path = (
        DATA_DIR
        / "calibration.parquet"
    )

    test_path = (
        DATA_DIR
        / "test.parquet"
    )

    calibration.to_parquet(
        calibration_path,
        index=False
    )

    test.to_parquet(
        test_path,
        index=False
    )

    print(
        f"\nCalibration dataset saved to: "
        f"{calibration_path}"
    )

    print(
        f"Test dataset saved to: "
        f"{test_path}"
    )

    # ---------------------------------------------------------
    # 9. Print summaries
    # ---------------------------------------------------------

    print_split_summary(
        "CALIBRATION SET",
        calibration
    )

    print_split_summary(
        "TEST SET",
        test
    )

    # ---------------------------------------------------------
    # 10. Save reproducibility manifest
    # ---------------------------------------------------------

    manifest = {
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "split_method": "StratifiedGroupKFold",
        "stratification": "dataset + subset",
        "group_column": "norm_key",
        "original_rows": int(len(df)),
        "duplicate_rows_removed": int(
            len(removed)
        ),
        "cleaned_rows": int(
            len(cleaned)
        ),
        "calibration_rows": int(
            len(calibration)
        ),
        "test_rows": int(
            len(test)
        ),
        "reference_conflict_rows": int(
            cleaned["reference_conflict"].sum()
        ),
        "files": {
            "unified":
                "data/unified_dataset.parquet",
            "calibration":
                "data/calibration.parquet",
            "test":
                "data/test.parquet",
            "removed_duplicates":
                "data/removed_duplicate_rows.parquet",
            "duplicate_groups":
                "data/duplicate_groups.csv",
        },
    }

    manifest_path = (
        DATA_DIR
        / "split_manifest.json"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2
        )

    print(
        f"\nSplit manifest saved to: "
        f"{manifest_path}"
    )

    print()
    print("=" * 70)
    print("A1 DATA FREEZE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()