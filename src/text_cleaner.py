"""Text cleaning utilities for transaction descriptions."""

import re
from typing import Any


def clean_description(text: Any) -> str:
    """Clean transaction description according to FR3 specification.

    Steps:
    1. Convert value to string.
    2. Convert to lowercase.
    3. Remove runs of four or more digits (transaction reference numbers).
    4. Replace punctuation and other non-alphanumeric characters with spaces.
    5. Collapse repeated whitespace.
    6. Trim leading and trailing spaces.

    Short digit sequences (< 4 digits) are preserved as they may represent
    legitimate merchant identifiers or store numbers.
    """
    if text is None:
        return ""
    
    text_str = str(text).lower()
    # Remove runs of 4 or more digits
    text_str = re.sub(r"\d{4,}", " ", text_str)
    # Replace punctuation and non-alphanumeric characters with spaces
    text_str = re.sub(r"[^a-zA-Z0-9\s]", " ", text_str)
    # Collapse repeated whitespace and trim
    text_str = re.sub(r"\s+", " ", text_str).strip()
    return text_str

