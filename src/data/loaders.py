"""
Unified loaders for HaluEval and TruthfulQA.

One row = one PROMPT.

This is the unit used for:
    - answer generation
    - duplicate detection
    - calibration/test splitting

Important:
    Labels belong to reference RESPONSES, not prompts.

    label = 0 -> reference response is correct / truthful / faithful
    label = 1 -> reference response is incorrect / untruthful / hallucinated

For the model's own generations in A2, labels are NOT assigned here.
They must be evaluated later against the reference information.

Datasets:
    HaluEval:
        - qa
        - dialogue
        - summarization

    TruthfulQA:
        - generation / validation
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Unified schema
# ---------------------------------------------------------------------------

SCHEMA_COLUMNS = [
    "uid",
    "dataset",
    "subset",
    "source_idx",
    "prompt",
    "context",
    "reference_correct",
    "reference_incorrect",
    "category",
    "label_semantics",
    "norm_key",
]


# ---------------------------------------------------------------------------
# Label semantics
# ---------------------------------------------------------------------------

LABEL_SEMANTICS = {
    "qa": (
        "HaluEval QA: a hallucinated answer contains information that "
        "contradicts or is unsupported by the provided knowledge. "
        "The reference hallucinated answer is an intentionally generated "
        "incorrect answer."
    ),
    "dialogue": (
        "HaluEval Dialogue: a hallucinated response contains facts that "
        "contradict or are unsupported by the provided knowledge and "
        "dialogue context. The reference hallucinated response is an "
        "intentionally generated incorrect response."
    ),
    "summarization": (
        "HaluEval Summarization: a hallucinated summary contains content "
        "that is not faithful to the source document. The reference "
        "hallucinated summary is an intentionally generated unfaithful "
        "summary."
    ),
    "truthfulqa": (
        "TruthfulQA: an incorrect answer is false or untruthful according "
        "to the dataset's truthfulness criterion, including answers that "
        "express common misconceptions or imitative falsehoods. "
        "Unlike HaluEval, this is not a grounded-context hallucination task."
    ),
}


# ---------------------------------------------------------------------------
# Expected raw columns
# ---------------------------------------------------------------------------

_HALUEVAL_COLUMNS = {
    "qa": {
        "knowledge",
        "question",
        "right_answer",
        "hallucinated_answer",
    },
    "dialogue": {
        "knowledge",
        "dialogue_history",
        "right_response",
        "hallucinated_response",
    },
    "summarization": {
        "document",
        "right_summary",
        "hallucinated_summary",
    },
}


_HALUEVAL_LOCAL_FILES = {
    "qa": "qa_data.json",
    "dialogue": "dialogue_data.json",
    "summarization": "summarization_data.json",
}


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(value: object) -> str:
    """
    Normalize text for duplicate / near-duplicate detection.

    This function is intentionally conservative.

    It:
        - converts Unicode to a canonical form
        - converts text to lowercase
        - normalizes whitespace
        - removes punctuation

    It does NOT remove non-English characters.

    Example:
        "  What is Paris?  "
        -> "what is paris"
    """

    if value is None:
        return ""

    text = str(value)

    # Normalize Unicode representation.
    text = unicodedata.normalize("NFKC", text)

    # Lowercase.
    text = text.lower().strip()

    # Replace punctuation/symbols with spaces.
    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        if category.startswith(("P", "S")):
            cleaned.append(" ")
        else:
            cleaned.append(char)

    text = "".join(cleaned)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_columns(
    df: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    """
    Check that a raw dataset contains all expected columns.
    """

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name}: missing expected columns "
            f"{sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )


def _require_non_empty_text(
    value: object,
    field_name: str,
    row_index: int,
) -> str:
    """
    Convert a value to string and ensure it is not empty.
    """

    if value is None:
        raise ValueError(
            f"Empty {field_name} at source row {row_index}"
        )

    # Only call pd.isna() for scalar-like values.
    if not isinstance(value, (list, tuple, np.ndarray)):
        try:
            if pd.isna(value):
                raise ValueError(
                    f"Empty {field_name} at source row {row_index}"
                )
        except (TypeError, ValueError):
            pass

    text = str(value).strip()

    if not text:
        raise ValueError(
            f"Empty {field_name} at source row {row_index}"
        )

    return text


def _as_list(value: object) -> list[str]:
    """
    Convert a reference-answer field into a clean list of strings.

    Handles:
        - Python lists
        - tuples
        - NumPy arrays
        - semicolon-separated strings
        - scalar values
        - missing values

    Important:
        NumPy arrays must be handled BEFORE pd.isna().
        Otherwise pd.isna(array) returns an array of booleans,
        which causes:

            ValueError:
            The truth value of an array with more than one element
            is ambiguous.
    """

    # ---------------------------------------------------------
    # 1. Missing None
    # ---------------------------------------------------------

    if value is None:
        return []

    # ---------------------------------------------------------
    # 2. NumPy arrays
    # ---------------------------------------------------------

    if isinstance(value, np.ndarray):
        value = value.tolist()

    # ---------------------------------------------------------
    # 3. Python lists / tuples
    # ---------------------------------------------------------

    if isinstance(value, (list, tuple)):
        values = value

    # ---------------------------------------------------------
    # 4. Strings
    # ---------------------------------------------------------

    elif isinstance(value, str):
        # Local CSV files may contain multiple answers separated
        # by semicolons.
        return [
            item.strip()
            for item in value.split(";")
            if item.strip()
        ]

    # ---------------------------------------------------------
    # 5. Scalar values / pandas missing values
    # ---------------------------------------------------------

    else:
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass

        values = [value]

    # ---------------------------------------------------------
    # 6. Clean individual values
    # ---------------------------------------------------------

    result = []

    for item in values:
        if item is None:
            continue

        # Skip missing scalar elements safely.
        if not isinstance(item, (list, tuple, np.ndarray)):
            try:
                if pd.isna(item):
                    continue
            except (TypeError, ValueError):
                pass

        text = str(item).strip()

        if text:
            result.append(text)

    return result


# ---------------------------------------------------------------------------
# HaluEval
# ---------------------------------------------------------------------------

def read_halueval(
    subset: str,
    source: str = "hf",
    local_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Read a raw HaluEval subset.

    Parameters
    ----------
    subset:
        One of:
            "qa"
            "dialogue"
            "summarization"

    source:
        "hf"    -> load from Hugging Face
        "local" -> load JSON Lines from local_dir

    local_dir:
        Directory containing:
            qa_data.json
            dialogue_data.json
            summarization_data.json

    Returns
    -------
    pandas.DataFrame
        Raw HaluEval data.
    """

    if subset not in _HALUEVAL_COLUMNS:
        raise ValueError(
            "subset must be one of "
            f"{list(_HALUEVAL_COLUMNS.keys())}"
        )

    if source == "local":

        if local_dir is None:
            raise ValueError(
                "local_dir must be provided when source='local'."
            )

        path = Path(local_dir) / _HALUEVAL_LOCAL_FILES[subset]

        if not path.exists():
            raise FileNotFoundError(
                f"HaluEval file not found: {path}"
            )

        return pd.read_json(path, lines=True)

    if source == "hf":

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Hugging Face datasets is required for "
                "source='hf'. Install it with:\n"
                "pip install datasets"
            ) from exc

        dataset = load_dataset(
            "pminervini/HaluEval",
            subset,
            split="data",
        )

        return dataset.to_pandas()

    raise ValueError(
        "source must be either 'hf' or 'local'."
    )


def normalize_halueval(
    raw: pd.DataFrame,
    subset: str,
) -> pd.DataFrame:
    """
    Convert one HaluEval subset into the unified schema.
    """

    if subset not in _HALUEVAL_COLUMNS:
        raise ValueError(
            "subset must be one of "
            f"{list(_HALUEVAL_COLUMNS.keys())}"
        )

    _check_columns(
        raw,
        _HALUEVAL_COLUMNS[subset],
        f"halueval/{subset}",
    )

    rows = []

    for i, record in enumerate(
        raw.to_dict(orient="records")
    ):

        if subset == "qa":

            prompt = _require_non_empty_text(
                record["question"],
                "question",
                i,
            )

            context = _require_non_empty_text(
                record["knowledge"],
                "knowledge",
                i,
            )

            correct = _require_non_empty_text(
                record["right_answer"],
                "right_answer",
                i,
            )

            incorrect = _require_non_empty_text(
                record["hallucinated_answer"],
                "hallucinated_answer",
                i,
            )

            # For QA, the question is the generation prompt.
            key_text = prompt

        elif subset == "dialogue":

            prompt = _require_non_empty_text(
                record["dialogue_history"],
                "dialogue_history",
                i,
            )

            context = _require_non_empty_text(
                record["knowledge"],
                "knowledge",
                i,
            )

            correct = _require_non_empty_text(
                record["right_response"],
                "right_response",
                i,
            )

            incorrect = _require_non_empty_text(
                record["hallucinated_response"],
                "hallucinated_response",
                i,
            )

            # Dialogue history identifies the prompt.
            key_text = prompt

        else:
            # Summarization:
            #
            # The model is asked to summarize the document.
            #
            # prompt  = instruction
            # context = source document
            #
            # We use the document for duplicate detection because
            # the instruction is identical for every row.

            prompt = "Summarize the following document."

            context = _require_non_empty_text(
                record["document"],
                "document",
                i,
            )

            correct = _require_non_empty_text(
                record["right_summary"],
                "right_summary",
                i,
            )

            incorrect = _require_non_empty_text(
                record["hallucinated_summary"],
                "hallucinated_summary",
                i,
            )

            key_text = context

        rows.append(
            {
                "uid": f"halueval-{subset}-{i:05d}",
                "dataset": "halueval",
                "subset": subset,
                "source_idx": i,
                "prompt": prompt,
                "context": context,
                "reference_correct": [correct],
                "reference_incorrect": [incorrect],
                "category": None,
                "label_semantics": LABEL_SEMANTICS[subset],
                "norm_key": normalize_text(key_text),
            }
        )

    return pd.DataFrame(
        rows,
        columns=SCHEMA_COLUMNS,
    )


# ---------------------------------------------------------------------------
# TruthfulQA
# ---------------------------------------------------------------------------

def read_truthfulqa(
    source: str = "hf",
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Read the TruthfulQA generation dataset.

    source="hf":
        Loads truthfulqa/truthful_qa, generation, validation.

    source="local":
        Loads a local CSV file.
    """

    if source == "local":

        if csv_path is None:
            raise ValueError(
                "csv_path must be provided when "
                "source='local'."
            )

        path = Path(csv_path)

        if not path.exists():
            raise FileNotFoundError(
                f"TruthfulQA CSV not found: {path}"
            )

        return pd.read_csv(path)

    if source == "hf":

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Hugging Face datasets is required for "
                "source='hf'. Install it with:\n"
                "pip install datasets"
            ) from exc

        dataset = load_dataset(
            "truthfulqa/truthful_qa",
            "generation",
            split="validation",
        )

        return dataset.to_pandas()

    raise ValueError(
        "source must be either 'hf' or 'local'."
    )


def normalize_truthfulqa(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert TruthfulQA generation data into the unified schema.
    """

    # Normalize column names while preserving the actual values.
    df = raw.rename(
        columns=lambda c:
        str(c).strip().lower().replace(" ", "_")
    )

    required = {
        "question",
        "best_answer",
        "correct_answers",
        "incorrect_answers",
    }

    _check_columns(
        df,
        required,
        "truthfulqa",
    )

    rows = []

    for i, record in enumerate(
        df.to_dict(orient="records")
    ):

        question = _require_non_empty_text(
            record["question"],
            "question",
            i,
        )

        correct = _as_list(
            record["correct_answers"]
        )

        incorrect = _as_list(
            record["incorrect_answers"]
        )

        best_answer = _require_non_empty_text(
            record["best_answer"],
            "best_answer",
            i,
        )

        # Make sure best_answer is included among
        # the correct reference answers.
        if best_answer not in correct:
            correct.insert(0, best_answer)

        if not correct:
            raise ValueError(
                f"TruthfulQA row {i} has no correct answers."
            )

        if not incorrect:
            raise ValueError(
                f"TruthfulQA row {i} has no incorrect answers."
            )

        category = record.get("category")

        if category is not None:
            try:
                if pd.isna(category):
                    category = None
            except (TypeError, ValueError):
                pass

        if category is not None:
            category = str(category).strip()

        rows.append(
            {
                "uid": f"truthfulqa-{i:05d}",
                "dataset": "truthfulqa",
                "subset": "truthfulqa",
                "source_idx": i,
                "prompt": question,
                "context": None,
                "reference_correct": correct,
                "reference_incorrect": incorrect,
                "category": category,
                "label_semantics": LABEL_SEMANTICS[
                    "truthfulqa"
                ],
                "norm_key": normalize_text(question),
            }
        )

    return pd.DataFrame(
        rows,
        columns=SCHEMA_COLUMNS,
    )


# ---------------------------------------------------------------------------
# Unified validation
# ---------------------------------------------------------------------------

def validate_unified(
    df: pd.DataFrame,
) -> None:
    """
    Validate the final unified prompt-level dataset.

    Raises ValueError when a structural problem is found.
    Prints useful warnings for duplicates because duplicate
    prompts are important for later split leakage checks.
    """

    # ---------------------------------------------------------
    # 1. Required columns
    # ---------------------------------------------------------

    missing = set(SCHEMA_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Unified dataset is missing columns: "
            f"{sorted(missing)}"
        )

    # ---------------------------------------------------------
    # 2. Empty prompts
    # ---------------------------------------------------------

    empty_prompts = (
        df["prompt"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_prompts.any():

        bad_indices = df.index[empty_prompts].tolist()

        raise ValueError(
            "Found empty prompts at rows: "
            f"{bad_indices[:10]}"
        )

    # ---------------------------------------------------------
    # 3. Empty normalized keys
    # ---------------------------------------------------------

    empty_keys = (
        df["norm_key"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_keys.any():

        bad_indices = df.index[empty_keys].tolist()

        raise ValueError(
            "Found empty norm_key values at rows: "
            f"{bad_indices[:10]}"
        )

    # ---------------------------------------------------------
    # 4. UID uniqueness
    # ---------------------------------------------------------

    duplicated_uids = df[
        df["uid"].duplicated(keep=False)
    ]

    if not duplicated_uids.empty:

        examples = duplicated_uids["uid"].tolist()[:10]

        raise ValueError(
            "Duplicate UIDs detected. "
            f"Examples: {examples}"
        )

    # ---------------------------------------------------------
    # 5. Valid dataset names
    # ---------------------------------------------------------

    valid_datasets = {
        "halueval",
        "truthfulqa",
    }

    invalid_datasets = set(
        df["dataset"].dropna().unique()
    ) - valid_datasets

    if invalid_datasets:

        raise ValueError(
            "Invalid dataset values: "
            f"{sorted(invalid_datasets)}"
        )

    # ---------------------------------------------------------
    # 6. Valid subsets
    # ---------------------------------------------------------

    valid_subsets = {
        "qa",
        "dialogue",
        "summarization",
        "truthfulqa",
    }

    invalid_subsets = set(
        df["subset"].dropna().unique()
    ) - valid_subsets

    if invalid_subsets:

        raise ValueError(
            "Invalid subset values: "
            f"{sorted(invalid_subsets)}"
        )

    # ---------------------------------------------------------
    # 7. Reference answers
    # ---------------------------------------------------------

    for index, row in df.iterrows():

        if not isinstance(
            row["reference_correct"],
            list,
        ):
            raise ValueError(
                f"reference_correct must be a list "
                f"at row {index}"
            )

        if not isinstance(
            row["reference_incorrect"],
            list,
        ):
            raise ValueError(
                f"reference_incorrect must be a list "
                f"at row {index}"
            )

        if len(row["reference_correct"]) == 0:
            raise ValueError(
                f"No correct reference answer "
                f"at row {index}"
            )

        if len(row["reference_incorrect"]) == 0:
            raise ValueError(
                f"No incorrect reference answer "
                f"at row {index}"
            )

    # ---------------------------------------------------------
    # 8. Duplicate normalized prompts
    # ---------------------------------------------------------

    duplicate_keys = df[
        df["norm_key"].duplicated(keep=False)
    ].sort_values("norm_key")

    if not duplicate_keys.empty:

        print(
            "\nWARNING: duplicate/identical normalized "
            "prompt keys detected."
        )

        print(
            "These must be handled carefully when "
            "creating calibration/test splits."
        )

        print(
            duplicate_keys[
                [
                    "uid",
                    "dataset",
                    "subset",
                    "norm_key",
                ]
            ].head(20).to_string(index=False)
        )

    print(
        f"\nValidation passed for {len(df):,} prompt rows."
    )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_all(
    source: str = "hf",
    halueval_dir: str | Path | None = None,
    truthfulqa_csv: str | Path | None = None,
    halueval_subsets: tuple[str, ...] = (
        "qa",
        "dialogue",
        "summarization",
    ),
) -> pd.DataFrame:
    """
    Load HaluEval + TruthfulQA into one unified schema.

    Parameters
    ----------
    source:
        "hf" or "local".

    halueval_dir:
        Required when source="local".

    truthfulqa_csv:
        Required when source="local".

    halueval_subsets:
        HaluEval subsets to load.

    Returns
    -------
    pandas.DataFrame
        One row per prompt.
    """

    # Validate requested HaluEval subsets first.
    invalid = set(halueval_subsets) - set(
        _HALUEVAL_COLUMNS.keys()
    )

    if invalid:
        raise ValueError(
            "Invalid HaluEval subsets: "
            f"{sorted(invalid)}"
        )

    parts = []

    # ---------------------------------------------------------
    # HaluEval
    # ---------------------------------------------------------

    for subset in halueval_subsets:

        raw = read_halueval(
            subset=subset,
            source=source,
            local_dir=halueval_dir,
        )

        normalized = normalize_halueval(
            raw,
            subset,
        )

        parts.append(normalized)

    # ---------------------------------------------------------
    # TruthfulQA
    # ---------------------------------------------------------

    raw_truthfulqa = read_truthfulqa(
        source=source,
        csv_path=truthfulqa_csv,
    )

    normalized_truthfulqa = normalize_truthfulqa(
        raw_truthfulqa,
    )

    parts.append(normalized_truthfulqa)

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    unified = pd.concat(
        parts,
        ignore_index=True,
    )

    # Make absolutely sure column order is correct.
    unified = unified[SCHEMA_COLUMNS]

    # ---------------------------------------------------------
    # Validate
    # ---------------------------------------------------------

    validate_unified(unified)

    return unified


# ---------------------------------------------------------------------------
# Candidate/reference-response view
# ---------------------------------------------------------------------------

def to_candidates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Explode reference answers into response-level rows.

    label:
        0 = correct / truthful / faithful
        1 = incorrect / untruthful / hallucinated

    IMPORTANT:
        These labels are for DATASET REFERENCE ANSWERS only.

        They must NOT be automatically assigned to A2
        generations.
    """

    rows = []

    for record in df.to_dict(orient="records"):

        # -----------------------------------------------------
        # Correct references
        # -----------------------------------------------------

        for response in record["reference_correct"]:

            rows.append(
                {
                    "uid": record["uid"],
                    "dataset": record["dataset"],
                    "subset": record["subset"],
                    "prompt": record["prompt"],
                    "context": record["context"],
                    "response": response,
                    "label": 0,
                    "reference_type": "correct",
                }
            )

        # -----------------------------------------------------
        # Incorrect references
        # -----------------------------------------------------

        for response in record["reference_incorrect"]:

            rows.append(
                {
                    "uid": record["uid"],
                    "dataset": record["dataset"],
                    "subset": record["subset"],
                    "prompt": record["prompt"],
                    "context": record["context"],
                    "response": response,
                    "label": 1,
                    "reference_type": "incorrect",
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "uid",
            "dataset",
            "subset",
            "prompt",
            "context",
            "response",
            "label",
            "reference_type",
        ],
    )


# ---------------------------------------------------------------------------
# Small convenience function for A1 reporting
# ---------------------------------------------------------------------------

def dataset_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a basic summary of the unified prompt dataset.

    Useful for the A1 data report.
    """

    summary = (
        df.groupby(
            ["dataset", "subset"],
            dropna=False,
        )
        .size()
        .reset_index(name="n_prompts")
    )

    return summary