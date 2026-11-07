"""Spending anomaly detection using robust Modified Z-Scores (median/MAD)."""

import numpy as np
import pandas as pd
from src import config


def calculate_modified_z_scores(amounts: np.ndarray) -> np.ndarray:
    """Compute modified z-scores for an array of transaction amounts.

    Uses the Median and Median Absolute Deviation (MAD):
    modified_z = 0.6745 * (amount - median) / MAD

    If MAD is 0, uses a floor: max(mean * 0.05, 1e-9).
    """
    if len(amounts) < config.MIN_ANOMALY_HISTORY:
        return np.zeros(len(amounts), dtype=float)

    median_val = float(np.median(amounts))
    abs_deviations = np.abs(amounts - median_val)
    mad_val = float(np.median(abs_deviations))

    if mad_val == 0.0:
        mean_val = float(np.mean(amounts))
        mad_val = max(mean_val * config.MAD_FLOOR_FACTOR, config.MAD_FLOOR_EPSILON)

    # FR5: Only unusually large amounts, not unusually small amounts
    scores = config.MODIFIED_Z_MULTIPLIER * (amounts - median_val) / mad_val
    return scores


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate anomaly scores and flags per category for transactions.

    Requirements (FR5):
    - Evaluated separately within each predicted category.
    - Category must contain at least 5 transactions; otherwise score is 0 and not flagged.
    - Flagged only when modified_z > 3.5.
    - Adds 'anomaly_score' (rounded to 2 decimal places) and 'is_anomaly' (boolean).
    """
    if df.empty:
        df_out = df.copy()
        df_out["anomaly_score"] = pd.Series(dtype=float)
        df_out["is_anomaly"] = pd.Series(dtype=bool)
        return df_out

    df_out = df.copy()
    df_out["anomaly_score"] = 0.0
    df_out["is_anomaly"] = False

    if "category" not in df_out.columns or "amount" not in df_out.columns:
        return df_out

    for category, group in df_out.groupby("category"):
        if len(group) >= config.MIN_ANOMALY_HISTORY:
            amounts = group["amount"].to_numpy(dtype=float)
            scores = calculate_modified_z_scores(amounts)
            scores_rounded = np.round(scores, 2)
            flags = scores > config.ANOMALY_MODIFIED_Z_THRESHOLD

            df_out.loc[group.index, "anomaly_score"] = scores_rounded
            df_out.loc[group.index, "is_anomaly"] = flags

    return df_out

