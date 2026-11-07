"""Data ingestion, validation, and parsing module."""

import io
import os
import re
from typing import Union, BinaryIO
import numpy as np
import pandas as pd

from src.config import MAX_UPLOAD_SIZE_BYTES, REQUIRED_COLUMNS


def parse_single_date(date_val: object) -> pd.Timestamp:
    """Parse a single date value, checking ISO format first, then day-first format.

    Returns pd.NaT if parsing fails.
    """
    if pd.isna(date_val):
        return pd.NaT

    date_str = str(date_val).strip()
    if not date_str:
        return pd.NaT

    # 1. Attempt strict/fast ISO parsing first (e.g. 2026-04-01 or 2026-04-01T...)
    try:
        parsed = pd.to_datetime(date_str, format="ISO8601")
        if pd.notna(parsed):
            return parsed.normalize()
    except (ValueError, TypeError):
        pass

    # 2. Attempt explicit standard ISO YYYY-MM-DD
    try:
        parsed = pd.to_datetime(date_str, format="%Y-%m-%d")
        if pd.notna(parsed):
            return parsed.normalize()
    except (ValueError, TypeError):
        pass

    # 3. Attempt day-first parsing (e.g. 01/04/2026 or 01-04-2026)
    try:
        parsed = pd.to_datetime(date_str, dayfirst=True)
        if pd.notna(parsed):
            return parsed.normalize()
    except (ValueError, TypeError):
        pass

    return pd.NaT


def parse_single_amount(amount_val: object) -> float:
    """Parse a single amount token, supporting currencies, commas, and negative signs.

    Extracts the first numeric token with optional sign and decimal places.
    Converts to absolute value as required by FR2.
    Returns np.nan if parsing fails or invalid.
    """
    if pd.isna(amount_val):
        return np.nan

    amount_str = str(amount_val).strip()
    if not amount_str:
        return np.nan

    # Remove commas
    cleaned = amount_str.replace(",", "")

    # Match first numeric token: optional minus/plus sign, digits, optional decimal digits
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        return np.nan

    try:
        val = float(match.group(0))
        # FR2: Convert amounts to absolute values
        abs_val = abs(val)
        return round(abs_val, 2)
    except (ValueError, OverflowError):
        return np.nan


def load_and_validate_csv(file_or_path: Union[str, BinaryIO, io.BytesIO, io.StringIO]) -> pd.DataFrame:
    """Load, validate, and parse a bank/UPI transaction CSV.

    Enforces:
    - Maximum upload size of 10 MB.
    - Required columns: date, description, amount (case-insensitive).
    - ISO and day-first date parsing.
    - Currency and negative sign amount extraction to positive float.
    - Dropping invalid dates, invalid amounts, and zero amounts.
    - Chronological sorting by date.
    - Rejection if no valid rows remain.

    Raises:
        ValueError: For any validation failure, file format error, or empty data.
    """
    # 1. Size and File Validation
    if isinstance(file_or_path, str):
        if not os.path.exists(file_or_path):
            raise ValueError(f"File not found: {file_or_path}")
        size = os.path.getsize(file_or_path)
        if size > MAX_UPLOAD_SIZE_BYTES:
            raise ValueError(
                f"File size ({size / (1024 * 1024):.1f} MB) exceeds maximum allowed size of 10 MB."
            )
        try:
            raw_df = pd.read_csv(file_or_path)
        except UnicodeDecodeError:
            try:
                raw_df = pd.read_csv(file_or_path, encoding="latin-1")
            except Exception as e:
                raise ValueError(f"Unable to read CSV file: {e}")
        except Exception as e:
            raise ValueError(f"Unable to read CSV file: {e}")
    else:
        # File-like object (e.g. Streamlit UploadedFile or BytesIO)
        # Check size if available
        size = getattr(file_or_path, "size", None)
        if size is None:
            # Check buffer size via tell/seek if supported
            try:
                curr = file_or_path.tell()
                file_or_path.seek(0, io.SEEK_END)
                size = file_or_path.tell()
                file_or_path.seek(curr)
            except Exception:
                size = 0

        if size and size > MAX_UPLOAD_SIZE_BYTES:
            raise ValueError(
                f"File size ({size / (1024 * 1024):.1f} MB) exceeds maximum allowed size of 10 MB."
            )

        try:
            raw_df = pd.read_csv(file_or_path)
        except UnicodeDecodeError:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            try:
                raw_df = pd.read_csv(file_or_path, encoding="latin-1")
            except Exception as e:
                raise ValueError(f"Unable to read CSV file: {e}")
        except Exception as e:
            raise ValueError(f"Unable to read CSV file: {e}")

    if raw_df is None or raw_df.empty:
        raise ValueError("Uploaded CSV file is empty.")

    # 2. Case-insensitive and trimmed column matching
    col_mapping = {}
    normalized_cols = {str(c).strip().lower(): c for c in raw_df.columns}

    missing_cols = []
    for req in REQUIRED_COLUMNS:
        if req in normalized_cols:
            col_mapping[normalized_cols[req]] = req
        else:
            missing_cols.append(req)

    if missing_cols:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing_cols)}. "
            f"CSV must contain: {', '.join(REQUIRED_COLUMNS)}."
        )

    # Filter and rename to canonical columns
    df = raw_df[list(col_mapping.keys())].rename(columns=col_mapping).copy()

    # 3. Parse Dates
    df["date"] = df["date"].apply(parse_single_date)

    # 4. Parse Amounts
    df["amount"] = df["amount"].apply(parse_single_amount)

    # 5. Drop invalid dates, invalid amounts, or zero amounts
    df = df.dropna(subset=["date", "amount"])
    df = df[df["amount"] > 0].copy()

    # Clean description column to string, strip whitespace
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df = df[df["description"] != ""].copy()

    if df.empty:
        raise ValueError(
            "No valid transactions remain after parsing dates and amounts. "
            "Please check that date formats and amounts are valid and non-zero."
        )

    # 6. Chronological Sort
    df = df.sort_values(by="date", ascending=True).reset_index(drop=True)

    # Keep only the required columns
    return df[["date", "description", "amount"]].copy()

