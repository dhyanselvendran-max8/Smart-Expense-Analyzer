"""Unit tests for the spending anomaly detector."""

import pandas as pd
import numpy as np
from src.anomaly_detector import calculate_modified_z_scores, detect_anomalies


def test_insufficient_history_not_scored():
    """Verify categories with < 5 transactions receive 0 score and are not flagged."""
    df = pd.DataFrame({
        "category": ["Food", "Food", "Food", "Food"],
        "amount": [100.0, 120.0, 110.0, 5000.0]  # Even with a huge value
    })
    res = detect_anomalies(df)
    assert (res["anomaly_score"] == 0.0).all()
    assert (res["is_anomaly"] == False).all()


def test_anomaly_detection_with_outlier():
    """Verify large outlier with >= 5 transactions is flagged."""
    # 7 regular Food transactions + 1 massive outlier
    df = pd.DataFrame({
        "category": ["Food"] * 8,
        "amount": [200.0, 220.0, 190.0, 210.0, 205.0, 195.0, 200.0, 5000.0]
    })
    res = detect_anomalies(df)
    assert not res.iloc[0]["is_anomaly"]
    # The last one should be flagged
    assert res.iloc[7]["is_anomaly"]
    assert res.iloc[7]["anomaly_score"] > 3.5


def test_zero_mad_floor_handling():
    """Verify MAD=0 case (all identical values + 1 outlier) uses floor and avoids zero-division."""
    # 5 identical amounts + 1 outlier
    df = pd.DataFrame({
        "category": ["Bills"] * 6,
        "amount": [500.0, 500.0, 500.0, 500.0, 500.0, 50000.0]
    })
    res = detect_anomalies(df)
    assert res.iloc[5]["is_anomaly"]
    assert res.iloc[5]["anomaly_score"] > 3.5


def test_empty_dataframe_anomaly():
    """Verify empty DataFrame returns required columns without error."""
    df = pd.DataFrame(columns=["category", "amount"])
    res = detect_anomalies(df)
    assert "anomaly_score" in res.columns
    assert "is_anomaly" in res.columns
    assert len(res) == 0

