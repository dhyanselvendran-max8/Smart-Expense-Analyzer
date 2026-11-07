"""Smart Expense Analyzer — Streamlit Web Application.

An Explainable Machine-Learning System for Personal Transaction Categorisation
and Spending Anomaly Detection.
"""

import io
import os
import streamlit as st
import pandas as pd

from src import config
from src.classifier import get_trained_classifier
from src.data_loader import load_and_validate_csv
from src.analysis import run_full_analysis

# 1. Page Configuration
st.set_page_config(
    page_title=config.APP_NAME,
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Session State Caching for ML Model
if "classifier" not in st.session_state:
    with st.spinner("Initializing and training transaction classification model..."):
        st.session_state["classifier"] = get_trained_classifier()

classifier = st.session_state["classifier"]

# 3. Sidebar Configuration
st.sidebar.title("💳 Navigation & Inputs")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Statement Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload bank/UPI statement (.csv)",
    type=["csv"],
    help="CSV file must contain date, description, and amount columns. Max size: 10 MB.",
)

# Convenience button to load bundled synthetic sample dataset
sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_transactions.csv")
use_sample = False
if os.path.exists(sample_path):
    if st.sidebar.button("📁 Load Synthetic Sample Dataset", use_container_width=True):
        st.session_state["use_sample_data"] = True

if st.session_state.get("use_sample_data", False):
    st.sidebar.info("Using bundled synthetic sample dataset.")
    if st.sidebar.button("Clear Sample Data", use_container_width=True):
        st.session_state["use_sample_data"] = False
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("2. Monthly budgets (Rs.)")
st.sidebar.caption("Set positive monthly budgets to track spending for the latest statement month.")

budgets = {}
for cat in config.CATEGORIES:
    budgets[cat] = st.sidebar.number_input(
        f"{cat} ({config.CURRENCY_SYMBOL})",
        min_value=0.0,
        step=500.0,
        value=0.0,
        format="%.2f",
        key=f"budget_{cat}",
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "🔒 **Privacy Notice**: All statements are parsed and analyzed purely within your local Python runtime. "
    "No financial data is sent to external servers or AI APIs."
)

# 4. Main Page Header
st.title(config.APP_NAME)
st.caption(config.APP_CAPTION)

# Determine data source
file_to_process = None
if uploaded_file is not None:
    # Clear sample flag if user uploaded a file
    st.session_state["use_sample_data"] = False
    file_to_process = uploaded_file
elif st.session_state.get("use_sample_data", False) and os.path.exists(sample_path):
    file_to_process = sample_path

# 5. Core Processing & Display
if file_to_process is None:
    st.info(
        "👋 Welcome! Please upload your bank or UPI statement CSV file using the sidebar, "
        "or click **'Load Synthetic Sample Dataset'** to preview the application with safe test data."
    )

    with st.expander("ℹ️ File Requirements & Supported Formats"):
        st.markdown(
            """
            - **File Type**: CSV (`.csv`), maximum size 10 MB.
            - **Required Columns** (case-insensitive):
                - `date`: ISO formats (`YYYY-MM-DD`) or day-first formats (`DD/MM/YYYY`, `DD-MM-YYYY`).
                - `description`: Transaction remarks or merchant details.
                - `amount`: Numeric value, optionally with currency prefix (`Rs. -1,200.50`, `₹ 450.00`).
            - **Privacy**: Processing is 100% offline and local.
            """
        )
else:
    try:
        # Load and validate CSV
        clean_df = load_and_validate_csv(file_to_process)

        # Run analysis pipeline
        result = run_full_analysis(clean_df, classifier, budgets)

        # Overview KPI Metric Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(
                label=f"Total Spend ({config.CURRENCY_SYMBOL})",
                value=f"₹{result.summary_stats['total_spend']:,.2f}",
            )
        with col2:
            st.metric(
                label="Transactions",
                value=result.summary_stats["transaction_count"],
            )
        with col3:
            st.metric(
                label="Anomalies Flagged",
                value=result.summary_stats["anomaly_count"],
            )
        with col4:
            st.metric(
                label="Probable Duplicates",
                value=result.summary_stats["duplicate_count"],
            )
        with col5:
            st.metric(
                label=f"Next Month Forecast ({config.CURRENCY_SYMBOL})",
                value=f"₹{result.forecast_amount:,.2f}",
            )

        st.markdown("---")

        # Five Required Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Transactions",
            "Monthly summary",
            "Anomalies",
            "Duplicates",
            "Budgets & forecast",
        ])

        # TAB 1: Transactions
        with tab1:
            st.subheader("All Categorised Transactions")
            st.caption("Transactions sorted chronologically with ML category predictions and confidence scores.")

            display_tx = result.transactions.copy()
            display_tx["date"] = pd.to_datetime(display_tx["date"]).dt.strftime("%Y-%m-%d")
            display_tx["amount"] = display_tx["amount"].apply(lambda v: f"₹{v:,.2f}")
            display_tx["confidence"] = display_tx["confidence"].apply(lambda v: f"{v:.3f}")

            # Column rename for display
            renamed_tx = display_tx.rename(columns={
                "date": "Date",
                "description": "Description",
                "amount": "Amount",
                "category": "Category",
                "confidence": "Confidence",
                "anomaly_score": "Anomaly Score",
                "is_anomaly": "Is Anomaly?",
                "is_duplicate": "Is Duplicate?",
            })

            st.dataframe(renamed_tx, use_container_width=True, hide_index=True)

            # Export Button
            csv_export_buffer = io.StringIO()
            result.transactions.to_csv(csv_export_buffer, index=False)
            st.download_button(
                label="📥 Download categorised CSV",
                data=csv_export_buffer.getvalue(),
                file_name="categorised.csv",
                mime="text/csv",
                help="Download full processed statement with categories, anomaly scores, and flags.",
            )

        # TAB 2: Monthly Summary
        with tab2:
            st.subheader("Monthly Category Spending")
            if not result.monthly_summary.empty:
                st.caption("Pivot table of aggregated monthly expenditures across categories (values in Rs.).")
                formatted_summary = result.monthly_summary.copy()
                st.dataframe(
                    formatted_summary.style.format("₹{:,.2f}"),
                    use_container_width=True,
                )

                st.subheader("Spending Distribution Chart")
                st.bar_chart(result.monthly_summary)
            else:
                st.info("No monthly summary data available.")

        # TAB 3: Anomalies
        with tab3:
            st.subheader("Spending Anomaly Detection")
            st.caption(
                "Unusually large expenses flagged using a robust modified Z-score (median & MAD) "
                "within each category (requires >= 5 transactions in category; threshold > 3.5)."
            )

            if not result.anomalies.empty:
                display_anomalies = result.anomalies.copy()
                display_anomalies["date"] = pd.to_datetime(display_anomalies["date"]).dt.strftime("%Y-%m-%d")
                display_anomalies["amount"] = display_anomalies["amount"].apply(lambda v: f"₹{v:,.2f}")
                display_anomalies["confidence"] = display_anomalies["confidence"].apply(lambda v: f"{v:.3f}")

                st.dataframe(
                    display_anomalies[
                        ["date", "description", "category", "amount", "anomaly_score", "confidence"]
                    ].rename(columns={
                        "date": "Date",
                        "description": "Description",
                        "category": "Category",
                        "amount": "Amount",
                        "anomaly_score": "Modified Z-Score",
                        "confidence": "Confidence",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No spending anomalies detected in the uploaded statement.")

        # TAB 4: Duplicates
        with tab4:
            st.subheader("Probable Duplicate Charges")
            st.caption(
                "Transactions having identical cleaned description and exact amount within a 2-day window. "
                "Note: This heuristic flags probable duplicates for user verification, not confirmed fraud."
            )

            if not result.duplicates.empty:
                display_dups = result.duplicates.copy()
                display_dups["date"] = pd.to_datetime(display_dups["date"]).dt.strftime("%Y-%m-%d")
                display_dups["amount"] = display_dups["amount"].apply(lambda v: f"₹{v:,.2f}")

                st.dataframe(
                    display_dups[["date", "description", "amount", "category"]].rename(columns={
                        "date": "Date",
                        "description": "Description",
                        "amount": "Amount",
                        "category": "Category",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No likely duplicate charges found.")

        # TAB 5: Budgets & Forecast
        with tab5:
            st.subheader("Budget Tracking")
            latest_label = result.latest_month or "N/A"
            st.caption(f"Comparing active category budgets against statement spending for latest month: **{latest_label}**.")

            if not result.budget_report.empty:
                formatted_budgets = result.budget_report.copy()
                formatted_budgets["budget"] = formatted_budgets["budget"].apply(lambda v: f"₹{v:,.2f}")
                formatted_budgets["spent"] = formatted_budgets["spent"].apply(lambda v: f"₹{v:,.2f}")
                formatted_budgets["percentage_used"] = formatted_budgets["percentage_used"].apply(lambda v: f"{v:.1f}%")

                st.dataframe(
                    formatted_budgets.rename(columns={
                        "category": "Category",
                        "budget": "Monthly Budget",
                        "spent": "Amount Spent",
                        "percentage_used": "% Used",
                        "status": "Status",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No active monthly budgets set. Enter positive budget amounts in the sidebar to track category targets.")

            st.markdown("---")
            st.subheader("Next-Month Expense Forecast")
            st.metric(
                label="Forecasted Next Month Spending",
                value=f"₹{result.forecast_amount:,.2f}",
            )
            st.caption(
                "⚠️ **Disclaimer**: The forecasted total is a simple moving average of the most recent 3 months "
                "(or available statement months). It is a rough baseline estimate and does not account for seasonality, "
                "trends, or one-time expenses."
            )

        # Footer Notes
        st.markdown("---")
        st.caption(
            "📌 **System Limitations**: Predictions are generated by a TF-IDF + Logistic Regression model trained on synthetic data. "
            "Displayed confidence scores do not guarantee correctness on unseen merchants. Always verify transactions against your official bank records."
        )

    except ValueError as err:
        st.error(f"⚠️ Validation Error: {err}")
    except Exception as ex:
        st.error(f"⚠️ An unexpected error occurred while processing the statement: {ex}")

