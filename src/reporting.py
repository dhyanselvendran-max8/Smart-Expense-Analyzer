"""Reporting, budget comparison, and expense forecasting module."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src import config


def generate_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate monthly pivot table of spending by category.

    Requirements (FR7):
    - Month derived as YYYY-MM.
    - Pivot table with months as rows and categories as columns.
    - Sum transaction amounts per month/category.
    - Fill missing combinations with 0.0.
    - Round totals to 2 decimal places.
    - Chronologically sorted by month.
    """
    if df.empty or "date" not in df.columns or "amount" not in df.columns:
        return pd.DataFrame(columns=config.CATEGORIES)

    df_copy = df.copy()
    df_copy["month"] = pd.to_datetime(df_copy["date"]).dt.strftime("%Y-%m")

    # Group and pivot
    pivot = df_copy.pivot_table(
        index="month",
        columns="category",
        values="amount",
        aggfunc="sum",
        fill_value=0.0,
    )

    # Ensure all standard categories exist as columns
    for cat in config.CATEGORIES:
        if cat not in pivot.columns:
            pivot[cat] = 0.0

    # Retain standard category order, followed by any other category (e.g., Uncategorised)
    extra_cols = [c for c in pivot.columns if c not in config.CATEGORIES]
    sorted_cols = [c for c in config.CATEGORIES if c in pivot.columns] + extra_cols
    pivot = pivot[sorted_cols]

    # Round to two decimal places
    pivot = pivot.round(2)
    return pivot.sort_index(ascending=True)


def calculate_budget_status(df: pd.DataFrame, budgets: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Evaluate spending against budgets for the latest month in the statement.

    Requirements (FR8):
    - Compare against the latest month in the statement.
    - For each active budget (budget > 0), show category, budget, spent, percentage_used, status.
    - Status is 'OVER' only when spending > non-zero budget; otherwise 'OK'.
    - Avoid division by zero.
    """
    if budgets is None:
        budgets = {}

    columns = ["category", "budget", "spent", "percentage_used", "status"]
    if df.empty or "date" not in df.columns or "amount" not in df.columns or "category" not in df.columns:
        return pd.DataFrame(columns=columns)

    df_copy = df.copy()
    df_copy["month"] = pd.to_datetime(df_copy["date"]).dt.strftime("%Y-%m")
    unique_months = sorted(df_copy["month"].unique())
    if not unique_months:
        return pd.DataFrame(columns=columns)

    latest_month = unique_months[-1]
    latest_df = df_copy[df_copy["month"] == latest_month]

    rows = []
    # Only report active (positive) budgets
    for cat in config.CATEGORIES:
        budget_amt = float(budgets.get(cat, 0.0))
        if budget_amt > 0:
            cat_spent = float(latest_df[latest_df["category"] == cat]["amount"].sum())
            pct_used = round((cat_spent / budget_amt) * 100.0, 1)
            status = "OVER" if cat_spent > budget_amt else "OK"
            rows.append({
                "category": cat,
                "budget": round(budget_amt, 2),
                "spent": round(cat_spent, 2),
                "percentage_used": pct_used,
                "status": status,
            })

    return pd.DataFrame(rows, columns=columns)


def calculate_next_month_forecast(df: pd.DataFrame) -> float:
    """Forecast next month's total spending using simple moving average.

    Requirements (FR9):
    - Sum all categories for each month's total spending.
    - Forecast next month's total as the mean of the most recent 3 monthly totals.
    - If fewer than 3 months exist, average available months.
    - If no monthly data exists, return 0.0.
    """
    if df.empty or "date" not in df.columns or "amount" not in df.columns:
        return 0.0

    df_copy = df.copy()
    df_copy["month"] = pd.to_datetime(df_copy["date"]).dt.strftime("%Y-%m")

    monthly_totals = df_copy.groupby("month")["amount"].sum().sort_index(ascending=True)
    if monthly_totals.empty:
        return 0.0

    # Most recent 3 months
    recent_totals = monthly_totals.tail(config.FORECAST_MOVING_AVERAGE_MONTHS)
    forecast_value = float(recent_totals.mean())
    return round(forecast_value, 2)

