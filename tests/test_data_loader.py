"""Unit tests for data loader and text cleaner modules."""

import io
import pytest
import pandas as pd
import numpy as np

from src.text_cleaner import clean_description
from src.data_loader import (
    parse_single_date,
    parse_single_amount,
    load_and_validate_csv,
)


def test_clean_description_basic():
    """Verify description cleaning rules from FR3."""
    assert clean_description("Swiggy Order #123456 UPI") == "swiggy order upi"
    assert clean_description("Domino's Pizza #12") == "domino s pizza 12"
    assert clean_description("  UBER   TRIP   DELHI  ") == "uber trip delhi"
    assert clean_description(None) == ""
    assert clean_description("") == ""


def test_clean_description_preserves_short_digits():
    """Verify 1-3 digit sequences are preserved for merchant names."""
    res = clean_description("24 Seven Convenience Store")
    assert "24" in res
    assert "seven" in res


def test_parse_single_date():
    """Verify ISO and day-first date parsing from FR2."""
    # ISO date
    d1 = parse_single_date("2026-04-01")
    assert d1 == pd.Timestamp("2026-04-01")

    # Day-first date (DD/MM/YYYY)
    d2 = parse_single_date("15/04/2026")
    assert d2 == pd.Timestamp("2026-04-15")

    # Day-first with dashes
    d3 = parse_single_date("25-12-2025")
    assert d3 == pd.Timestamp("2025-12-25")

    # Invalid
    assert pd.isna(parse_single_date("not-a-date"))
    assert pd.isna(parse_single_date(None))


def test_parse_single_amount():
    """Verify currency, comma, and negative amount parsing from FR2."""
    # Currency with comma and negative sign
    assert parse_single_amount("Rs. -1,200.50") == 1200.50
    assert parse_single_amount("-500.00") == 500.00
    assert parse_single_amount("₹ 3,450.75") == 3450.75
    assert parse_single_amount("100") == 100.00
    assert parse_single_amount("0") == 0.00
    assert np.isnan(parse_single_amount("invalid"))
    assert np.isnan(parse_single_amount(None))


def test_load_and_validate_csv_success():
    """Verify valid CSV loading, case-insensitive headers, and sorting."""
    csv_data = """Date,Description,Amount
2026-04-05,Zomato Food Order,Rs. -450.00
01/04/2026,Uber Ride Bengaluru,180.50
2026-04-03,Amazon Shopping,1250.00
"""
    buffer = io.StringIO(csv_data)
    df = load_and_validate_csv(buffer)

    assert len(df) == 3
    assert list(df.columns) == ["date", "description", "amount"]
    # Check chronological order: 01/04/2026, 2026-04-03, 2026-04-05
    assert df.iloc[0]["amount"] == 180.50
    assert df.iloc[1]["amount"] == 1250.00
    assert df.iloc[2]["amount"] == 450.00


def test_load_and_validate_csv_missing_column():
    """Verify missing required columns raise descriptive ValueError."""
    csv_data = """Date,Amount
2026-04-01,100
"""
    buffer = io.StringIO(csv_data)
    with pytest.raises(ValueError, match="Missing required column"):
        load_and_validate_csv(buffer)


def test_load_and_validate_csv_filters_invalid_and_zero():
    """Verify zero and unparseable amounts/dates are dropped."""
    csv_data = """date,description,amount
2026-04-01,Valid Transaction,150.00
2026-04-02,Zero Amount,0
invalid-date,Broken Date,200.00
2026-04-03,Broken Amount,invalid
"""
    buffer = io.StringIO(csv_data)
    df = load_and_validate_csv(buffer)
    assert len(df) == 1
    assert df.iloc[0]["description"] == "Valid Transaction"


def test_load_and_validate_csv_empty_after_filter():
    """Verify rejection when no valid transactions remain."""
    csv_data = """date,description,amount
invalid,No valid data,0
"""
    buffer = io.StringIO(csv_data)
    with pytest.raises(ValueError, match="No valid transactions remain"):
        load_and_validate_csv(buffer)


def test_load_and_validate_csv_size_exceeded():
    """Verify file exceeding size limit is rejected."""
    # Create dummy buffer with simulated size > 10MB
    class LargeBuffer(io.StringIO):
        @property
        def size(self):
            return 11 * 1024 * 1024

    buf = LargeBuffer("date,description,amount\n2026-04-01,test,10\n")
    with pytest.raises(ValueError, match="exceeds maximum allowed size"):
        load_and_validate_csv(buf)

