"""Analysis orchestration module coordinating end-to-end pipeline execution."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd

from src.classifier import TransactionClassifier
from src.anomaly_detector import detect_anomalies
from src.duplicate_detector import mark_duplicates
from src.reporting import (
    generate_monthly_summary,
    calculate_budget_status,
    calculate_next_month_forecast,
)


@dataclass
class AnalysisResult:
    """Container holding all analytical products from statement processing."""
    transactions: pd.DataFrame
    monthly_summary: pd.DataFrame
    budget_report: pd.DataFrame
    forecast_amount: float
    latest_month: Optional[str]
    anomalies: pd.DataFrame
    duplicates: pd.DataFrame
    summary_stats: Dict[str, Any]


def run_full_analysis(
    df: pd.DataFrame,
    classifier: TransactionClassifier,
    budgets: Optional[Dict[str, float]] = None,
) -> AnalysisResult:
    """Run full analysis pipeline on parsed transactions.

    Executes in sequence:
    1. Transaction categorisation & confidence scoring (ML)
    2. Category-wise spending anomaly detection (MAD modified Z-score)
    3. Probable duplicate charge detection (heuristic)
    4. Monthly category spending summary & pivot
    5. Active budget evaluation on latest statement month
    6. Moving average next-month expense forecast

    Returns an AnalysisResult containing all outputs.
    """
    if budgets is None:
        budgets = {}

    if df.empty:
        empty_tx = pd.DataFrame(
            columns=["date", "description", "amount", "category", "confidence", "anomaly_score", "is_anomaly", "is_duplicate"]
        )
        return AnalysisResult(
            transactions=empty_tx,
            monthly_summary=pd.DataFrame(),
            budget_report=pd.DataFrame(),
            forecast_amount=0.0,
            latest_month=None,
            anomalies=pd.DataFrame(),
            duplicates=pd.DataFrame(),
            summary_stats={
                "total_spend": 0.0,
                "transaction_count": 0,
                "anomaly_count": 0,
                "duplicate_count": 0,
            },
        )

    # 1. Categorisation
    categorised_df = classifier.predict_dataframe(df)

    # 2. Anomaly Detection
    with_anomalies_df = detect_anomalies(categorised_df)

    # 3. Duplicate Detection
    processed_df = mark_duplicates(with_anomalies_df)

    # 4. Monthly Summary Pivot Table
    monthly_summary = generate_monthly_summary(processed_df)

    # 5. Budget Report
    budget_report = calculate_budget_status(processed_df, budgets)

    # 6. Next-Month Forecast
    forecast_amount = calculate_next_month_forecast(processed_df)

    # Latest month identifier
    months = pd.to_datetime(processed_df["date"]).dt.strftime("%Y-%m").unique()
    latest_month = sorted(months)[-1] if len(months) > 0 else None

    # Filtered views
    anomalies_df = processed_df[processed_df["is_anomaly"]].copy()
    duplicates_df = processed_df[processed_df["is_duplicate"]].copy()

    # Summary metrics
    summary_stats = {
        "total_spend": round(float(processed_df["amount"].sum()), 2),
        "transaction_count": len(processed_df),
        "anomaly_count": len(anomalies_df),
        "duplicate_count": len(duplicates_df),
        "uncategorised_count": len(processed_df[processed_df["category"] == "Uncategorised"]),
    }

    return AnalysisResult(
        transactions=processed_df,
        monthly_summary=monthly_summary,
        budget_report=budget_report,
        forecast_amount=forecast_amount,
        latest_month=latest_month,
        anomalies=anomalies_df,
        duplicates=duplicates_df,
        summary_stats=summary_stats,
    )

