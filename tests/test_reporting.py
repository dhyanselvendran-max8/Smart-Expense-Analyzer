"""Unit tests for reporting, budgeting, forecasting, and analysis orchestration."""

import pandas as pd
from src import config
from src.classifier import get_trained_classifier
from src.reporting import (
    generate_monthly_summary,
    calculate_budget_status,
    calculate_next_month_forecast,
)
from src.analysis import run_full_analysis


def test_generate_monthly_summary():
    """Verify pivot table structure, filling missing categories with 0, and rounding."""
    df = pd.DataFrame({
        "date": [
            pd.Timestamp("2026-01-10"),
            pd.Timestamp("2026-01-20"),
            pd.Timestamp("2026-02-05"),
        ],
        "category": ["Food", "Transport", "Food"],
        "amount": [250.50, 120.00, 310.25],
    })

    summary = generate_monthly_summary(df)

    assert "2026-01" in summary.index
    assert "2026-02" in summary.index
    assert summary.loc["2026-01", "Food"] == 250.50
    assert summary.loc["2026-01", "Transport"] == 120.00
    assert summary.loc["2026-01", "Rent"] == 0.00  # unfilled standard category
    assert summary.loc["2026-02", "Food"] == 310.25
    assert summary.loc["2026-02", "Transport"] == 0.00


def test_calculate_budget_status():
    """Verify budget comparison against latest month and OK/OVER status."""
    df = pd.DataFrame({
        "date": [
            pd.Timestamp("2026-01-15"),  # Earlier month
            pd.Timestamp("2026-02-10"),  # Latest month
            pd.Timestamp("2026-02-15"),  # Latest month
        ],
        "category": ["Food", "Food", "Entertainment"],
        "amount": [1500.0, 3000.0, 1500.0],
    })

    budgets = {
        "Food": 2500.0,           # Spent is 3000.0 -> OVER
        "Entertainment": 2000.0,  # Spent is 1500.0 -> OK
        "Transport": 0.0,         # Inactive (0) -> not in report
    }

    report = calculate_budget_status(df, budgets)

    assert len(report) == 2
    food_row = report[report["category"] == "Food"].iloc[0]
    assert food_row["spent"] == 3000.0
    assert food_row["budget"] == 2500.0
    assert food_row["percentage_used"] == 120.0
    assert food_row["status"] == "OVER"

    ent_row = report[report["category"] == "Entertainment"].iloc[0]
    assert ent_row["spent"] == 1500.0
    assert ent_row["budget"] == 2000.0
    assert ent_row["percentage_used"] == 75.0
    assert ent_row["status"] == "OK"


def test_calculate_next_month_forecast():
    """Verify 3-month moving average forecast rules."""
    # Empty
    assert calculate_next_month_forecast(pd.DataFrame()) == 0.0

    # 2 months: averages both (1000 + 2000)/2 = 1500
    df_2m = pd.DataFrame({
        "date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")],
        "amount": [1000.0, 2000.0],
    })
    assert calculate_next_month_forecast(df_2m) == 1500.0

    # 4 months: averages latest 3 months (2000 + 3000 + 4000)/3 = 3000.0
    df_4m = pd.DataFrame({
        "date": [
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-02-01"),
            pd.Timestamp("2026-03-01"),
            pd.Timestamp("2026-04-01"),
        ],
        "amount": [1000.0, 2000.0, 3000.0, 4000.0],
    })
    assert calculate_next_month_forecast(df_4m) == 3000.0


def test_run_full_analysis():
    """Verify full end-to-end orchestration pipeline."""
    classifier = get_trained_classifier()

    df = pd.DataFrame({
        "date": [
            pd.Timestamp("2026-04-01"),
            pd.Timestamp("2026-04-02"),
            pd.Timestamp("2026-04-02"),
            pd.Timestamp("2026-04-03"),
            pd.Timestamp("2026-04-04"),
            pd.Timestamp("2026-04-05"),
        ],
        "description": [
            "Swiggy Order UPI 111",
            "Swiggy Order UPI 222",  # Duplicate candidate within 1 day, same amount
            "Uber Ride Airport",
            "BESCOM Electricity Utility",
            "Netflix Streaming",
            "Swiggy Order Dinner 333",
        ],
        "amount": [300.0, 300.0, 250.0, 1200.0, 649.0, 400.0],
    })

    budgets = {"Food": 1500.0, "Bills": 1000.0}
    res = run_full_analysis(df, classifier, budgets)

    assert len(res.transactions) == 6
    assert "category" in res.transactions.columns
    assert "anomaly_score" in res.transactions.columns
    assert "is_duplicate" in res.transactions.columns
    assert res.summary_stats["transaction_count"] == 6
    assert res.latest_month == "2026-04"
    assert res.forecast_amount > 0.0

