# Smart Expense Analyzer

**An Explainable Machine-Learning System for Personal Transaction Categorisation and Spending Anomaly Detection**

An educational prototype and local personal finance analysis tool built with Python, Streamlit, Pandas, NumPy, and Scikit-learn.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Tech Stack](#architecture--tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Running the Test Suite](#running-the-test-suite)
- [Using the Synthetic Sample Dataset](#using-the-synthetic-sample-dataset)
- [Privacy & Security Guarantees](#privacy--security-guarantees)
- [System Limitations](#system-limitations)

---

## Overview

The **Smart Expense Analyzer** ingests bank and UPI transaction statements in CSV format, validates and parses statement dates and amounts, categorises transactions into 8 standard classes using a TF-IDF + Logistic Regression model, flags unusual spending using robust modified Z-scores (median/MAD), detects probable duplicate charges, and computes monthly pivot summaries, category budget tracking, and next-month expense forecasts.

All processing occurs **strictly locally on your computer**. No financial statements or personal transaction records are ever sent to remote servers or third-party AI APIs.

---

## Key Features

1. **CSV Ingestion & Robust Parsing (FR1, FR2)**
   - Case-insensitive matching for required columns (`date`, `description`, `amount`).
   - Supports both ISO dates (`YYYY-MM-DD`) and day-first dates (`DD/MM/YYYY`, `DD-MM-YYYY`).
   - Parses currency formatting, commas, and negative signs (e.g., `Rs. -1,200.50`, `₹ 450.00`) and converts amounts to absolute positive values.
   - Enforces a 10 MB maximum file size limit.
   - Drops invalid dates, zero amounts, and unparseable rows without exposing Python stack traces.

2. **Description Cleaning (FR3)**
   - Strips transaction reference numbers (runs of 4+ digits).
   - Preserves short numbers (< 4 digits) for merchant names (e.g., `24 Seven`).
   - Replaces punctuation with spaces, collapses whitespace, and lowercases text.

3. **Machine Learning Categorisation (FR4)**
   - Categorises transactions across 8 predefined classes:
     - **Food**, **Transport**, **Shopping**, **Bills**, **Entertainment**, **Health**, **Education**, **Rent**
   - Feature extraction uses `FeatureUnion`:
     - Word TF-IDF: unigrams + bigrams (`ngram_range=(1, 2)`), `sublinear_tf=True`.
     - Character TF-IDF: within word boundaries (`char_wb`), `ngram_range=(3, 5)`, `sublinear_tf=True`.
   - Classification via `LogisticRegression(C=8.0, max_iter=1000)`.
   - Confidence thresholding: Predictions with highest probability < 0.35 are marked **Uncategorised**.

4. **Spending Anomaly Detection (FR5)**
   - Robust category-level outlier detection using Modified Z-Scores:
     $$\text{Modified Z} = \frac{0.6745 \times (\text{amount} - \text{median})}{\text{MAD}}$$
   - Handles zero-MAD scenarios with a fallback floor: $\max(\text{mean} \times 0.05, 10^{-9})$.
   - Requires at least 5 transactions in a category before calculating scores.
   - Flags transactions where Modified Z > 3.5 (only unusually large amounts).

5. **Probable Duplicate Detection (FR6)**
   - Flags charges having identical cleaned descriptions and identical amounts within a 2-day calendar window.
   - Heuristic alert intended for user verification.

6. **Monthly Aggregation, Budgets & Forecasting (FR7, FR8, FR9)**
   - Pivot table matrix displaying monthly category totals (in Rs.).
   - Category budget tracking against the statement's latest month with `OK` / `OVER` indicators.
   - Next-month forecast computed using a 3-month moving average of total expenditures.

7. **Export & UI (FR10)**
   - Clean 5-tab Streamlit interface (`Transactions`, `Monthly summary`, `Anomalies`, `Duplicates`, `Budgets & forecast`).
   - One-click download of processed data as `categorised.csv`.

---

## Architecture & Tech Stack

- **Python**: 3.10+
- **Streamlit**: Web dashboard and presentation layer
- **Pandas**: CSV parsing, aggregation, grouping, pivot tables
- **NumPy**: Numerical operations and Median Absolute Deviation (MAD) calculations
- **Scikit-learn**: TF-IDF vectorization, FeatureUnion, and Logistic Regression
- **Pytest**: Automated unit testing

---

## Project Structure

```text
expens/
│
├── app.py                          # Streamlit application layout & UI tabs
├── requirements.txt                # Core and testing dependencies
├── README.md                       # Documentation & operating manual
├── .gitignore                      # Git exclusion rules
│
├── data/
│   └── sample_transactions.csv     # Safe synthetic sample dataset
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration constants & vocabularies
│   ├── text_cleaner.py             # clean_description() logic
│   ├── data_loader.py              # File ingestion, validation, and parsing
│   ├── classifier.py               # Synthetic training dataset & ML pipeline
│   ├── anomaly_detector.py         # Modified Z-score (median/MAD) detection
│   ├── duplicate_detector.py       # Heuristic duplicate detection (<= 2 days)
│   ├── reporting.py                # Pivot tables, budget checks, moving-average forecast
│   └── analysis.py                 # Pipeline orchestrator
│
└── tests/
    ├── __init__.py
    ├── test_config.py              # Configuration specification tests
    ├── test_data_loader.py         # File ingestion & cleaning tests
    ├── test_classifier.py          # ML pipeline & categorisation tests
    ├── test_anomaly_detector.py    # Anomaly detection & MAD tests
    ├── test_duplicate_detector.py  # Duplicate charge heuristic tests
    └── test_reporting.py           # Aggregations, budgets & forecast tests
```

---

## Installation & Setup

1. **Clone or open the workspace:**
   ```bash
   cd /path/to/expens
   ```

2. **Create and activate a Python virtual environment (recommended):**
   - On Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - On Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

---

## Running the Application

Launch the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## Running the Test Suite

Run the full automated pytest suite:

```bash
python -m pytest -v
```

All 30 unit tests cover:
- Category names, merchant seed vocabularies, and threshold constants
- Text cleaning rules and short-digit preservation
- Date parsing (ISO and day-first formats)
- Currency token extraction (e.g. `Rs. -1,200.50` -> `1200.50`)
- File size rejection (> 10MB) and missing column errors
- Deterministic synthetic dataset generation (960 rows)
- Machine learning categorisation, confidence scores, and Uncategorised fallback
- Anomaly scoring, MAD floor handling, and minimum transaction history constraints
- Duplicate charge detection and date-window boundaries
- Monthly pivot tables, latest-month budget checks, and moving-average forecasts

---

## Using the Synthetic Sample Dataset

The repository includes a ready-to-use synthetic statement at `data/sample_transactions.csv`.

- **To preview without uploading:** Click the **"📁 Load Synthetic Sample Dataset"** button in the sidebar.
- **To test manual upload:** Upload `data/sample_transactions.csv` using the file uploader in the sidebar.

The sample dataset exercises:
- Multi-month timeline (Feb 2026, Mar 2026, Apr 2026) for forecasting.
- Multiple date formats (ISO `2026-04-01` and Day-First `03/04/2026`).
- Currency prefixes and negative amounts (`Rs. -290.00`, `Rs. -1,850.00`).
- An intentional duplicate transaction within 2 days (Swiggy Food Delivery on 03/04 and 04/04).
- High-value anomalies (Swiggy Catering Order in Food; Croma Laptop in Shopping).

---

## Privacy & Security Guarantees

- **No Remote Transmission**: All CSV parsing, text processing, and model inference run locally in your Python process.
- **No External APIs or Telemetry**: No LLM APIs, cloud databases, or tracking services are used.
- **Safe Testing**: Never upload real banking credentials or statements to shared public repositories. The `.gitignore` prevents accidental commit of transaction CSVs while whitelisting `data/sample_transactions.csv`.

---

## System Limitations

As detailed in the project specification and evaluation report:
1. **Synthetic Training Baseline**: The classifier is trained on a synthetic seed dataset. Unseen or novel merchants may be misclassified or labeled as `Uncategorised`.
2. **Uncalibrated Confidence**: A high confidence score indicates proximity to learned merchant templates, not a 100% guarantee of correctness.
3. **Anomaly Scoring Heuristic**: Categories with fewer than 5 transactions are not evaluated for anomalies. The Modified Z-score flags large outliers relative to median spending in that category.
4. **Duplicate Heuristic**: Identical amounts and descriptions within 2 days are flagged for user review. This is not proof of fraud and may flag legitimate recurring purchases.
5. **Moving-Average Forecast**: Next-month projection is a 3-month simple moving average. It does not account for seasonality, one-off spikes, or trends.
6. **Session-Only Budgets**: Budgets are entered during the session and are not stored in a persistent database.

