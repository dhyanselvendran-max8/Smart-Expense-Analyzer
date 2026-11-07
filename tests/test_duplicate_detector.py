"""Unit tests for the duplicate transaction detector."""

import pandas as pd
from src.duplicate_detector import mark_duplicates, find_duplicates


def test_duplicate_within_two_days():
    """Verify identical cleaned description + exact amount within 2 days is flagged."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-02")],
        "description": ["Swiggy Order #9910", "Swiggy Order #8820"],  # Cleaned both become 'swiggy order'
        "amount": [450.00, 450.00]
    })
    res = mark_duplicates(df)
    assert not res.iloc[0]["is_duplicate"]
    assert res.iloc[1]["is_duplicate"]

    dups = find_duplicates(df)
    assert len(dups) == 1
    assert dups.iloc[0]["description"] == "Swiggy Order #8820"


def test_repeated_charge_month_later_not_duplicate():
    """Verify monthly subscription repetition (> 2 days) is NOT flagged as duplicate."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-01")],
        "description": ["Netflix Streaming Subscription", "Netflix Streaming Subscription"],
        "amount": [649.00, 649.00]
    })
    res = mark_duplicates(df)
    assert not res["is_duplicate"].any()


def test_different_amount_same_day_not_duplicate():
    """Verify different amount on same day is NOT flagged as duplicate."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-01")],
        "description": ["Uber Ride", "Uber Ride"],
        "amount": [150.00, 320.00]
    })
    res = mark_duplicates(df)
    assert not res["is_duplicate"].any()


def test_different_description_same_amount_not_duplicate():
    """Verify different descriptions with same amount are NOT flagged."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-01")],
        "description": ["Amazon Shopping", "Zomato Food"],
        "amount": [500.00, 500.00]
    })
    res = mark_duplicates(df)
    assert not res["is_duplicate"].any()


def test_empty_dataframe_duplicates():
    """Verify empty DataFrame handled gracefully."""
    df = pd.DataFrame(columns=["date", "description", "amount"])
    res = mark_duplicates(df)
    assert "is_duplicate" in res.columns
    assert len(res) == 0

