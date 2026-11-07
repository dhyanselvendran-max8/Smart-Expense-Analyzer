"""Unit tests for the transaction classification model."""

import pytest
import pandas as pd
from src import config
from src.classifier import (
    generate_synthetic_training_data,
    build_pipeline,
    TransactionClassifier,
    get_trained_classifier,
)


def test_generate_synthetic_training_data():
    """Verify synthetic dataset size, distribution, and determinism."""
    descriptions, labels = generate_synthetic_training_data(seed=config.RANDOM_SEED)

    assert len(descriptions) == 960
    assert len(labels) == 960

    # Exactly 120 per category
    counts = pd.Series(labels).value_counts().to_dict()
    for cat in config.CATEGORIES:
        assert counts[cat] == 120

    # Deterministic reproducibility check
    d2, l2 = generate_synthetic_training_data(seed=config.RANDOM_SEED)
    assert descriptions == d2
    assert labels == l2


def test_classifier_training_and_known_merchants():
    """Verify classifier trains and predicts known seed merchants accurately."""
    clf = get_trained_classifier()
    assert clf.model is not None
    assert set(clf.model.classes_) == set(config.CATEGORIES)

    # Test Food
    cat, conf = clf.predict_single("Swiggy Delivery UPI 12345678 Bangalore")
    assert cat == "Food"
    assert conf >= config.CONFIDENCE_THRESHOLD

    # Test Transport
    cat, conf = clf.predict_single("Uber Trip Airport ride POS")
    assert cat == "Transport"
    assert conf >= config.CONFIDENCE_THRESHOLD

    # Test Entertainment
    cat, conf = clf.predict_single("Netflix subscription streaming")
    assert cat == "Entertainment"
    assert conf >= config.CONFIDENCE_THRESHOLD

    # Test Rent
    cat, conf = clf.predict_single("Apartment Monthly Rent transfer landlord")
    assert cat == "Rent"
    assert conf >= config.CONFIDENCE_THRESHOLD


def test_classifier_empty_and_uncategorised():
    """Verify empty text or ambiguous low-confidence input falls back to Uncategorised."""
    clf = get_trained_classifier()

    # Empty text
    cat, conf = clf.predict_single("")
    assert cat == config.UNCATEGORISED_LABEL
    assert conf == 0.0

    # Random noisy gibberish with no match to any merchant or vocabulary
    cat, conf = clf.predict_single("zzqxkj plmvbr tzqwx")
    assert cat == config.UNCATEGORISED_LABEL
    assert conf < config.CONFIDENCE_THRESHOLD


def test_predict_dataframe():
    """Verify DataFrame batch prediction adds category and confidence columns."""
    clf = get_trained_classifier()

    df = pd.DataFrame({
        "date": [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-02")],
        "description": ["Swiggy Order Food", "Unknown random gibberish xyz123"],
        "amount": [350.0, 100.0]
    })

    result_df = clf.predict_dataframe(df)
    assert "category" in result_df.columns
    assert "confidence" in result_df.columns
    assert len(result_df) == 2
    assert result_df.iloc[0]["category"] == "Food"
    assert isinstance(result_df.iloc[0]["confidence"], float)


def test_predict_dataframe_empty():
    """Verify empty DataFrame prediction handles gracefully."""
    clf = get_trained_classifier()
    empty_df = pd.DataFrame(columns=["date", "description", "amount"])
    result = clf.predict_dataframe(empty_df)
    assert "category" in result.columns
    assert "confidence" in result.columns
    assert len(result) == 0

