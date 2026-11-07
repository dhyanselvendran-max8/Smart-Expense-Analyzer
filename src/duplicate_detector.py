"""Duplicate charge detection based on cleaned descriptions, amounts, and dates."""

import pandas as pd
from src import config
from src.text_cleaner import clean_description


def mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Identify probable duplicate charges and add 'is_duplicate' boolean column.

    Requirements (FR6):
    - Cleaned description matches previous transaction's cleaned description.
    - Amount exactly matches previous transaction's amount.
    - Date gap between transactions is <= 2 days.
    - Sorted by cleaned description, amount, date before comparing adjacent rows.
    - Flags the later matching row(s) as duplicates.
    """
    if df.empty:
        df_out = df.copy()
        df_out["is_duplicate"] = pd.Series(dtype=bool)
        return df_out

    df_out = df.copy()
    original_order = df_out.index

    # Add temporary cleaned description for comparison
    df_out["_cleaned_desc"] = df_out["description"].apply(clean_description)
    df_out["_date_parsed"] = pd.to_datetime(df_out["date"])
    df_out["is_duplicate"] = False

    # Sort by cleaned description, amount, and date
    sorted_df = df_out.sort_values(
        by=["_cleaned_desc", "amount", "_date_parsed"]
    ).reset_index(drop=False)  # keep 'index' referencing original position

    duplicate_indices = set()
    n = len(sorted_df)

    for i in range(1, n):
        curr_row = sorted_df.iloc[i]
        prev_row = sorted_df.iloc[i - 1]

        # Check cleaned description match
        if curr_row["_cleaned_desc"] == prev_row["_cleaned_desc"] and curr_row["_cleaned_desc"] != "":
            # Check exact amount match
            if curr_row["amount"] == prev_row["amount"]:
                # Check date difference <= 2 days
                days_diff = (curr_row["_date_parsed"] - prev_row["_date_parsed"]).days
                if 0 <= days_diff <= config.DUPLICATE_MAX_DAYS_GAP:
                    duplicate_indices.add(curr_row["index"])

    df_out.loc[list(duplicate_indices), "is_duplicate"] = True
    df_out = df_out.drop(columns=["_cleaned_desc", "_date_parsed"])
    return df_out.reindex(original_order)


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return only transactions flagged as probable duplicates."""
    df_marked = mark_duplicates(df)
    return df_marked[df_marked["is_duplicate"]].copy()

