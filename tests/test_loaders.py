import numpy as np
import pandas as pd
import pytest

from src.data.loaders import (
    SCHEMA_COLUMNS,
    normalize_halueval,
    normalize_truthfulqa,
    to_candidates,
)

from src.data.report import auroc, build_report


# ---------------------------------------------------------------------------
# Small synthetic fixtures
# ---------------------------------------------------------------------------

QA = pd.DataFrame(
    {
        "knowledge": ["k1", "k2"],
        "question": [
            "Who wrote X?",
            "Who wrote X?",
        ],
        "right_answer": [
            "Alice",
            "Bob",
        ],
        "hallucinated_answer": [
            "Carol is the author",
            "Dan",
        ],
    }
)


DLG = pd.DataFrame(
    {
        "knowledge": ["k"],
        "dialogue_history": [
            "[Human]: hi"
        ],
        "right_response": [
            "hello there"
        ],
        "hallucinated_response": [
            "greetings"
        ],
    }
)


SUM = pd.DataFrame(
    {
        "document": [
            "doc text"
        ],
        "right_summary": [
            "short"
        ],
        "hallucinated_summary": [
            "a longer wrong one"
        ],
    }
)


TQA = pd.DataFrame(
    {
        "type": [
            "Adversarial"
        ],
        "category": [
            "Misconceptions"
        ],
        "question": [
            "Do we use 10% of our brains?"
        ],
        "best_answer": [
            "No"
        ],
        "correct_answers": [
            ["No", "We use all of it"]
        ],
        "incorrect_answers": [
            ["Yes", "Only 10%"]
        ],
        "source": [
            "s"
        ],
    }
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def frame() -> pd.DataFrame:
    """
    Build a small unified dataframe containing all supported subsets.
    """

    return pd.concat(
        [
            normalize_halueval(QA, "qa"),
            normalize_halueval(DLG, "dialogue"),
            normalize_halueval(SUM, "summarization"),
            normalize_truthfulqa(TQA),
        ],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_schema_and_uids():
    """
    Every normalized dataframe must have the exact unified schema,
    and every prompt must have a unique UID.
    """

    df = frame()

    assert list(df.columns) == SCHEMA_COLUMNS
    assert df["uid"].is_unique


def test_expected_subset_values():
    """
    Check that every supported subset appears correctly.
    """

    df = frame()

    assert set(df["dataset"]) == {
        "halueval",
        "truthfulqa",
    }

    assert set(df["subset"]) == {
        "qa",
        "dialogue",
        "summarization",
        "truthfulqa",
    }


# ---------------------------------------------------------------------------
# HaluEval tests
# ---------------------------------------------------------------------------

def test_halueval_qa_normalization():
    """
    Check that QA fields map to the unified schema correctly.
    """

    df = normalize_halueval(QA, "qa")

    assert df.loc[0, "dataset"] == "halueval"
    assert df.loc[0, "subset"] == "qa"

    assert df.loc[0, "prompt"] == "Who wrote X?"
    assert df.loc[0, "context"] == "k1"

    assert df.loc[0, "reference_correct"] == [
        "Alice"
    ]

    assert df.loc[0, "reference_incorrect"] == [
        "Carol is the author"
    ]


def test_halueval_dialogue_normalization():
    """
    Check dialogue-specific field mapping.
    """

    df = normalize_halueval(
        DLG,
        "dialogue",
    )

    assert df.loc[0, "prompt"] == "[Human]: hi"
    assert df.loc[0, "context"] == "k"

    assert df.loc[0, "reference_correct"] == [
        "hello there"
    ]

    assert df.loc[0, "reference_incorrect"] == [
        "greetings"
    ]


def test_halueval_summarization_normalization():
    """
    Summarization uses a fixed instruction as prompt
    and the document as context.
    """

    df = normalize_halueval(
        SUM,
        "summarization",
    )

    assert (
        df.loc[0, "prompt"]
        == "Summarize the following document."
    )

    assert df.loc[0, "context"] == "doc text"

    assert df.loc[0, "reference_correct"] == [
        "short"
    ]

    assert df.loc[0, "reference_incorrect"] == [
        "a longer wrong one"
    ]


def test_missing_column_raises():
    """
    Missing required HaluEval columns must raise ValueError.
    """

    broken = QA.drop(
        columns=["knowledge"]
    )

    with pytest.raises(ValueError):
        normalize_halueval(
            broken,
            "qa",
        )


def test_invalid_halueval_subset_raises():
    """
    Unknown HaluEval subsets must be rejected.
    """

    with pytest.raises(ValueError):
        normalize_halueval(
            QA,
            "invalid_subset",
        )


# ---------------------------------------------------------------------------
# TruthfulQA tests
# ---------------------------------------------------------------------------

def test_truthfulqa_refs_and_candidates():
    """
    TruthfulQA should preserve multiple correct/incorrect references.
    """

    df = normalize_truthfulqa(TQA)

    assert df.loc[0, "reference_correct"] == [
        "No",
        "We use all of it",
    ]

    assert df.loc[0, "reference_incorrect"] == [
        "Yes",
        "Only 10%",
    ]

    candidates = to_candidates(df)

    assert candidates["label"].tolist() == [
        0,
        0,
        1,
        1,
    ]


def test_truthfulqa_best_answer_is_added():
    """
    best_answer must be included in reference_correct
    even if it was absent from correct_answers.
    """

    data = TQA.copy()

    data["correct_answers"] = [
        ["We use all of it"]
    ]

    df = normalize_truthfulqa(data)

    assert df.loc[0, "reference_correct"] == [
        "No",
        "We use all of it",
    ]


# ---------------------------------------------------------------------------
# Candidate tests
# ---------------------------------------------------------------------------

def test_candidates_preserve_uid():
    """
    Exploding references must preserve the original prompt UID.
    """

    df = normalize_halueval(
        QA,
        "qa",
    )

    candidates = to_candidates(df)

    assert set(candidates["uid"]) == set(
        df["uid"]
    )


def test_candidates_have_only_binary_labels():
    """
    Reference-response labels must only be 0 or 1.
    """

    candidates = to_candidates(frame())

    assert set(candidates["label"]) <= {
        0,
        1,
    }


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def test_duplicate_prompt_detected_in_report():
    """
    The two QA rows intentionally have the same question.
    The report should identify duplicated normalized prompts.
    """

    rep = build_report(frame())

    assert "rows_sharing_a_duplicate_prompt" in rep

    # QA contains two rows with the same normalized prompt.
    # Therefore both rows should be counted as duplicate rows.
    assert "| qa | 2 |" in rep


# ---------------------------------------------------------------------------
# AUROC tests
# ---------------------------------------------------------------------------

def test_auroc_perfect_ordering():
    """
    Perfectly separated scores should produce AUROC = 1.0.
    """

    scores = np.array([
        1,
        2,
        3,
        4,
    ])

    labels = np.array([
        0,
        0,
        1,
        1,
    ])

    assert auroc(
        scores,
        labels,
    ) == pytest.approx(1.0)


def test_auroc_random_tie_case():
    """
    If every prediction score is identical, AUROC should be 0.5.
    """

    scores = np.array([
        1,
        1,
        1,
        1,
    ])

    labels = np.array([
        0,
        0,
        1,
        1,
    ])

    assert auroc(
        scores,
        labels,
    ) == pytest.approx(0.5)