Antigravity Build Guide — Smart Expense Analyzer
Source: DS FINAL REPORT (1).pdf (Mini Project Report, October 2026)
Project title: Smart Expense Analyzer
Subtitle: An Explainable Machine-Learning System for Personal Transaction Categorisation and Spending Anomaly Detection

This document converts the supplied project report into an implementation brief and a step-by-step workflow for Google Antigravity. Treat the report as the project specification. Do not silently add features that are outside its stated scope.
1. What you are building
Build a local, browser-based Python application that accepts a user's bank/UPI transaction statement in CSV format, categorises each transaction using machine learning, detects unusually large transactions and probable duplicates, and displays spending summaries, budget checks, and a basic next-month forecast.

The application is an educational prototype. It is not a banking application, financial adviser, fraud detector, or production-grade financial product.
Main user workflow
1. The user opens the local Streamlit application.
2. The user uploads a CSV statement.
3. The application validates the file and parses its date, description, and amount columns.
4. Transaction descriptions are cleaned and categorised by a TF-IDF + Logistic Regression model.
5. Low-confidence predictions are marked Uncategorised.
6. The application calculates anomaly scores and checks for possible duplicate charges.
7. It displays transactions, monthly category totals, anomalies, duplicates, budgets, and a forecast.
8. The user can download the categorised transactions as a CSV.
Privacy requirement: process uploaded statements locally. The core workflow must not call external AI APIs or send transaction data to a remote service.
2. Required technology
Use the stack specified in the report:
- Python 3.10 or later
- Streamlit — user interface
- pandas — CSV parsing, data cleaning, grouping, and pivot tables
- NumPy — numerical operations
- scikit-learn — TF-IDF and Logistic Regression
Install dependencies:
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scikit-learn
Create a requirements.txt containing:
streamlit
pandas
numpy
scikit-learn
Do not add a database, cloud backend, login system, OCR, PDF parsing, or bank integration in the first version. These are outside the report's implemented scope.
3. Functional requirements
Implement all of the following.
FR1 — CSV upload and validation
- Accept .csv files only.
- Maximum upload size: 10 MB.
- Require these columns (case-insensitive after trimming and lowercasing):
  - date
  - description
  - amount
- Report missing columns clearly.
- Handle unreadable or malformed CSV files without showing a Python traceback to the user.
- Reject a file if no valid transactions remain after parsing.
FR2 — Date and amount parsing
- Parse ISO dates first, such as 2026-04-01.
- If ISO parsing fails for a value, attempt day-first parsing, such as 01/04/2026.
- Remove commas from amount strings.
- Extract the first numeric token, including an optional minus sign and decimal part. This should support values such as Rs. -1,200.50.
- Convert amounts to absolute values.
- Drop rows with invalid dates, invalid amounts, or zero amounts.
- Sort the final transactions by date.
- Keep only the required columns for the core pipeline unless additional source columns are deliberately preserved and documented.
FR3 — Description cleaning
Create a reusable clean_description(text) function:
1. Convert the value to a string.
2. Convert it to lowercase.
3. Remove runs of four or more digits (typically transaction reference numbers).
4. Replace punctuation and other non-alphanumeric characters with spaces.
5. Collapse repeated whitespace.
6. Trim leading and trailing spaces.
Keep short digit strings because they may be part of a merchant name.
FR4 — Transaction categorisation
Use exactly these eight categories:
1. Food
2. Transport
3. Shopping
4. Bills
5. Entertainment
6. Health
7. Education
8. Rent
Build a synthetic starter dataset with 120 examples per category, for 960 labelled rows in total. The report describes 10 merchant phrases per category, with random payment-channel noise, reference numbers, and city names.

Use a scikit-learn pipeline with two combined TF-IDF feature extractors:
- Word TF-IDF:
  - ngram_range=(1, 2) (unigrams and bigrams)
  - sublinear_tf=True
  - Use clean_description as the preprocessor.
- Character TF-IDF:
  - analyzer="char_wb"
  - ngram_range=(3, 5)
  - sublinear_tf=True
  - Use clean_description as the preprocessor.
Combine the two feature matrices using make_union, then train:
- LogisticRegression(max_iter=1000, C=8.0)
Use a fixed random seed of 7 when generating the synthetic seed dataset so the initial model is reproducible.

For each transaction:
- Obtain class probabilities using predict_proba.
- Select the class with the highest probability.
- Treat the highest probability as the displayed confidence.
- If confidence is below 0.35, return Uncategorised instead of forcing a category.
- Round displayed confidence to three decimal places.
Important: the report's evaluation shows that confidence is not reliably calibrated and unseen merchants can be misclassified with high confidence. Show the confidence, but do not present it as a guarantee of correctness.
FR5 — Spending anomaly detection
Use a robust modified z-score separately within each predicted category.

For a category with at least 5 transactions:
- median = median(amounts)
- MAD = median(abs(amounts - median))
- If MAD is zero, use a small floor: max(mean(amounts) * 0.05, 1e-9).
- Calculate:
modified_z = 0.6745 * (amount - median) / MAD
- Flag a transaction only when modified_z > 3.5.
The report intentionally flags unusually large amounts, not unusually small amounts. Categories with fewer than five transactions must not receive anomaly scores; leave their scores at zero and do not flag them.

Expose the numeric anomaly score in the transaction data and display flagged rows in the Anomalies tab.
FR6 — Probable duplicate detection
A transaction is a probable duplicate when:
- Its cleaned description matches the previous transaction's cleaned description.
- Its amount exactly matches the previous transaction's amount.
- The date gap is no more than 2 days.
Sort by cleaned description, amount, and date before comparing adjacent rows. Return the later matching row(s) as probable duplicates.

This is a heuristic, not proof of fraud. Legitimate repeated purchases within two days can be flagged. A monthly subscription should not be flagged merely because it repeats a month later.
FR7 — Monthly spending summary
- Derive a YYYY-MM month value from each transaction date.
- Create a pivot table with months as rows and categories as columns.
- Sum transaction amounts for each month/category combination.
- Fill missing combinations with zero.
- Round totals to two decimal places.
- Display the result as a table and a bar chart.
FR8 — Budget checks
- Provide one user-editable monthly budget input for each of the eight categories.
- Use rupees (Rs.) in labels.
- Compare budgets against the latest month in the uploaded statement.
- For each active budget, show category, budget, amount spent, percentage used, and status.
- Status is OVER only when spending is greater than a non-zero budget; otherwise show OK.
- Avoid division by zero.
- Budgets may be session-only; persistent storage is not part of the original prototype.
FR9 — Next-month forecast
- Sum all categories to get each month's total spending.
- Forecast next month's total as the mean of the most recent three monthly totals.
- If fewer than three months exist, average the available months.
- If no monthly data exists, return 0.0.
- Label this clearly as a rough moving-average estimate, not a reliable prediction.
FR10 — Export
Provide a download button for the categorised transaction data as categorised.csv.

Include useful computed columns such as category, confidence, anomaly score, and anomaly flag.
4. User interface specification
Use a clean, readable Streamlit layout. The report specifies a wide page layout and a sidebar.
Sidebar
- CSV file uploader.
- Heading: Monthly budgets (Rs.).
- One numeric budget input per category.
- Default budget can be zero; only positive budgets need to be checked.
Main page
- Title: Smart Expense Analyzer.
- Caption explaining that the user uploads a CSV with date, description, and amount, and that processing is local.
- Before a file is uploaded, show a clear instruction to upload a statement.
- After a valid upload, show exactly these five tabs:
1. Transactions
   - Display the full categorised transaction table.
   - Include confidence and anomaly information.
   - Include the Download categorised CSV button.
2. Monthly summary
   - Display monthly totals by category.
   - Include a bar chart.
3. Anomalies
   - Display only transactions where is_anomaly is true.
   - Make the anomaly score visible.
   - If no anomalies exist, show a clear empty state.
4. Duplicates
   - Display probable duplicate rows.
   - If none are found, show No likely duplicate charges found.
5. Budgets & forecast
   - Display budget status for active budgets.
   - Display the next-month forecast in rupees.
   - Explain that the forecast is a simple moving average.
UI and error-handling expectations
- Use st.set_page_config(page_title=APP_NAME, layout="wide").
- Use st.session_state to train/cache the default model once per session rather than retraining on every interaction.
- Show validation errors using st.error(...).
- Do not expose stack traces to the user.
- Use readable column names, sensible currency formatting, and empty-state messages.
- Keep the interface functional at ordinary laptop screen sizes.
5. Recommended project structure
The report's prototype uses one Python file, but for maintainability, implement the same logic in separate modules while preserving the report's behaviour.
smart-expense-analyzer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── text_cleaner.py
│   ├── classifier.py
│   ├── anomaly_detector.py
│   ├── duplicate_detector.py
│   ├── reporting.py
│   └── analysis.py
│
└── tests/
    ├── test_data_loader.py
    ├── test_classifier.py
    ├── test_anomaly_detector.py
    ├── test_duplicate_detector.py
    └── test_reporting.py
Module responsibilities
File	Responsibility
config.py	App name, eight categories, thresholds, upload limit, seed merchants, noise words
data_loader.py	CSV size check, reading, column validation, date/amount parsing
text_cleaner.py	clean_description()
classifier.py	Synthetic training-data generation, TF-IDF pipeline, model training, categorisation
anomaly_detector.py	Median/MAD modified z-score and anomaly flag
duplicate_detector.py	Same cleaned description + amount within two days
reporting.py	Monthly summary, budget checks, moving-average forecast
analysis.py	Orchestrate the full analysis pipeline and return results
app.py	Streamlit interface only
tests/	Automated tests for the core logic


Keep core data-processing functions independent of Streamlit so they can be tested without launching the UI. Import Streamlit in app.py, not throughout the core modules.
6. Step-by-step build instructions for Antigravity
Use Antigravity's project/workspace mode and open a new folder named smart-expense-analyzer.
Step 1 — Give Antigravity the project brief
1. Create the project folder.
2. Save this document as ANTIGRAVITY_BUILD_GUIDE.md in the root folder.
3. Open that folder as the workspace in Antigravity.
4. Ask the agent to read this entire Markdown file before writing code.
5. Tell it to implement the report's requirements without inventing extra features.
Initial prompt to paste into Antigravity:
Read ANTIGRAVITY_BUILD_GUIDE.md completely. It is the source of truth for this project.

Build the Smart Expense Analyzer as a local Python + Streamlit application.
Follow the specified features, categories, thresholds, parsing rules, algorithms,
UI tabs, and limitations exactly.

First inspect the workspace and propose a short implementation plan.
Then create the project structure and implement the application in small,
verifiable stages. Do not generate the entire project as one untested block.
Do not add cloud services, external AI APIs, bank integrations, PDF/OCR support,
authentication, or a database. Keep transaction data local.

After each stage:
1. Explain which files were created or changed.
2. Run relevant tests or checks.
3. Fix errors before proceeding.
4. Do not claim a feature works unless you have tested it.

Do not use real bank statements for tests. Generate synthetic sample data.
Step 2 — Create the project structure and configuration
Ask Antigravity:
Implement Step 1 of the build: create the folder structure from
ANTIGRAVITY_BUILD_GUIDE.md and implement src/config.py.

Define the exact eight categories, confidence threshold 0.35, anomaly
threshold 3.5, minimum anomaly history 5, maximum upload size 10 MB,
random seed 7, merchant seed vocabulary, and payment-noise vocabulary.

Do not implement the UI yet. Show me the created files and verify imports.
Check that the eight category names and constants match this guide.
Step 3 — Implement CSV ingestion
Implement src/data_loader.py according to the guide.

Requirements:
- CSV only, maximum 10 MB.
- Required columns: date, description, amount (case-insensitive).
- Parse ISO dates first, then day-first dates.
- Parse amounts such as Rs. -1,200.50 correctly.
- Convert amounts to absolute values.
- Remove invalid dates, invalid amounts, and zero amounts.
- Raise clear ValueError messages for invalid files or no remaining rows.
- Return transactions sorted by date.

Create and run tests for valid input, missing columns, invalid rows,
DD/MM/YYYY dates, ISO dates, and currency-formatted negative amounts.
Do not continue until the tests pass.
Step 4 — Implement text cleaning and the classifier
Implement src/text_cleaner.py and src/classifier.py.

Follow the exact model design in ANTIGRAVITY_BUILD_GUIDE.md:
- 120 synthetic examples for each of eight classes (960 total).
- Fixed random seed 7.
- Word TF-IDF unigrams and bigrams, sublinear_tf=True.
- Character TF-IDF using char_wb, n-grams 3 to 5, sublinear_tf=True.
- Combine with make_union.
- LogisticRegression(max_iter=1000, C=8.0).
- Use predict_proba and label predictions below 0.35 as Uncategorised.
- Return category and confidence rounded to three decimals.

Keep the synthetic dataset generation deterministic. Add tests for a known
seed merchant, empty input, and low-confidence/unknown text. Do not claim
the model is accurate on real bank statements.
Step 5 — Implement anomaly and duplicate detection
Implement src/anomaly_detector.py and src/duplicate_detector.py.

Anomalies:
- Group by predicted category.
- Require at least five transactions in a category.
- Use median and MAD.
- If MAD is zero, use max(mean * 0.05, 1e-9).
- Modified z-score = 0.6745 * (amount - median) / MAD.
- Flag only scores greater than 3.5.

Duplicates:
- Clean the description.
- Require identical cleaned description and identical amount.
- Compare adjacent rows after sorting by cleaned description, amount, date.
- Date gap must be at most two days.

Add tests for normal transactions, large outliers, insufficient history,
zero MAD, matching duplicates, and repeated charges more than two days apart.
Explain that these are heuristics, not proof of fraud.
Step 6 — Implement reporting and analysis orchestration
Implement src/reporting.py and src/analysis.py.

Reporting must provide:
- Monthly pivot table: month rows, category columns, summed amounts,
  missing combinations filled with zero, two-decimal rounding.
- Budget comparison against the latest month only.
- Budget status OK or OVER and percentage used.
- Forecast = mean of the latest three monthly total-spend values
  (or fewer if fewer months exist); return 0.0 for empty data.

The analysis module must run categorisation, anomaly scoring, duplicate
detection, monthly aggregation, and forecast, then return a clear result
object or documented dictionary.

Add unit tests for empty inputs, multiple months, budget boundaries,
and forecast calculations.
Step 7 — Build the Streamlit interface
Implement app.py using the interface specification in
ANTIGRAVITY_BUILD_GUIDE.md.

Use a wide layout, a sidebar CSV uploader, one rupee budget input per
category, and exactly five tabs:
Transactions, Monthly summary, Anomalies, Duplicates, Budgets & forecast.

Cache the trained default model in st.session_state so it is not retrained
on every rerun. Display validation errors cleanly. Add the categorised CSV
download button. Include empty states and a disclaimer that users should
verify results against the original statement.

Do not add features outside the report. Run the app and fix UI/runtime errors.
Step 8 — Generate synthetic sample data
Ask Antigravity to create a safe sample file:
Create data/sample_transactions.csv with synthetic transaction data only.
Include date, description, and amount columns. Include examples from all
eight categories, different date formats if supported by the loader,
currency-formatted amounts, at least one likely duplicate within two days,
and enough transactions in a category to exercise anomaly detection.

Inject a few clearly unusual large amounts for testing. Do not use any real
person's bank data, account number, UPI ID, or transaction reference.
Document that the data is synthetic.
If you create a data/ folder, add it to the project tree and decide whether sample data should be committed. Never put a real statement in the repository.
Step 9 — Run the application
Open a terminal in the project root and run:
python -m venv .venv
Activate the environment on Windows PowerShell:
.venv\Scripts\Activate.ps1
If PowerShell blocks activation, use Command Prompt:
.venv\Scripts\activate.bat
Install dependencies:
python -m pip install -r requirements.txt
Run tests:
python -m pytest -q
If pytest is not included in requirements.txt, install it for development:
python -m pip install pytest
Launch Streamlit:
python -m streamlit run app.py
Open the local URL printed in the terminal (normally http://localhost:8501). Upload data/sample_transactions.csv and test each tab.
Step 10 — Perform the final acceptance check
Do not call the project complete until all checks below pass.
- [ ] App starts without errors.
- [ ] CSV upload works.
- [ ] Missing required columns produce a readable error.
- [ ] Files larger than 10 MB are rejected.
- [ ] ISO and DD/MM/YYYY dates parse correctly.
- [ ] Rs. -1,200.50 parses to 1200.50.
- [ ] Invalid and zero-amount rows are removed.
- [ ] All eight categories are available.
- [ ] Known seed merchants are categorised plausibly.
- [ ] Confidence is displayed and values below 0.35 become Uncategorised.
- [ ] Categories with fewer than five transactions are not anomaly-scored.
- [ ] Large outliers above the modified z-score threshold are flagged.
- [ ] Same-description, same-amount transactions within two days are listed as probable duplicates.
- [ ] Monthly category totals are correct.
- [ ] Budgets compare against the latest statement month.
- [ ] Forecast uses the latest three monthly totals, or fewer if necessary.
- [ ] Categorised CSV download works.
- [ ] Empty states and invalid inputs do not crash the app.
- [ ] Core processing does not send statement data to external services.
- [ ] Automated tests pass.
7. Required tests and honest performance reporting
The report's original test results are reference results from its synthetic test setup, not guaranteed results for a newly implemented version:
Test in the report	Reported result	Correct interpretation
Synthetic held-out classifier split	100.0% accuracy; macro F1 1.00	Optimistic because train/test examples share generated merchant templates
48 unseen hand-written descriptions	58.3% raw top-class accuracy	Shows weak generalisation to unseen merchants
Unseen descriptions with 0.35 threshold	56.2% final-label accuracy; 14 uncategorised	Threshold avoids some forced guesses, but does not eliminate wrong predictions
Synthetic anomaly test	7 of 8 injected outliers found; 0 false positives	Result applies to the specific synthetic test
Functional tests	11 of 11 passed	The report's original prototype passed its listed cases
174-row analysis runtime	0.036 seconds	One measurement on the report's development machine, excluding model training


Do not copy these numbers into a new test report unless the new implementation has actually reproduced them using the same test data and procedure. Run the tests and report the new results honestly.
Important limitations to show in the app or README
- The classifier starts from a synthetic merchant vocabulary.
- New merchants may be miscategorised or marked Uncategorised.
- A high confidence score does not guarantee a correct prediction.
- Anomaly detection can miss smaller outliers or outliers in naturally variable categories.
- Categories with fewer than five transactions are not anomaly-scored.
- Duplicate detection can flag legitimate repeated purchases within two days.
- The moving-average forecast ignores trend, seasonality, and one-off expenses.
- Only CSV is supported; no PDF statements, OCR, or direct bank connection.
- Budgets and user corrections are not persisted between sessions.
- No real customer statements or user study were used in the report's evaluation.
8. Scope boundaries: do not implement these in the first version
The report lists these as future work, not current features:
- Training on real, consented, user-corrected transactions.
- A correction interface with incremental model retraining.
- Probability calibration.
- Comparing Logistic Regression with Naive Bayes, linear SVM, or embedding classifiers.
- Per-class evaluation on a real labelled dataset.
- Per-merchant and seasonal anomaly baselines.
- More advanced time-series forecasting.
- Recurring-payment detection.
- PDF statement parsing or OCR.
- Persistent encrypted storage and user accounts.
- Tamil and English multilingual merchant handling.
- A supervised usability study.
Keep these out of the first build unless the project owner explicitly changes the scope.
9. Security and responsible-use requirements
- Use synthetic data for demonstrations and automated tests.
- Do not upload real bank statements to public/shared deployments.
- Do not log full transaction descriptions or sensitive statement contents.
- Do not send transaction data to third-party APIs.
- Make it clear that the user must verify flagged transactions against the original bank statement.
- Do not describe a duplicate flag as confirmed fraud.
- Do not use the system for credit decisions, surveillance, or decisions about another person without their knowledge.
- If the application is ever hosted, reassess access control, encryption, logs, file retention, and applicable data-protection requirements before deployment.
10. Final instruction to Antigravity
After implementation, ask Antigravity to:
1. Inspect every project file for missing imports, inconsistent function names, and duplicated logic.
2. Run the full automated test suite.
3. Launch the Streamlit application and resolve startup/runtime errors.
4. Test the synthetic CSV through all five tabs.
5. Verify the downloadable CSV contents.
6. Update README.md with installation, activation, run, test, and privacy instructions.
7. Provide a concise final report listing implemented features, test results, known limitations, and any incomplete items.
Definition of done: the app runs locally, all specified features work against synthetic test data, the tests pass, and the implementation does not claim reliability beyond the evidence.
11. Source report notes
This build guide is based on the supplied 25-page report, especially:
- Pages 3–5: abstract, motivation, problem statement, objectives, scope, and workflow.
- Pages 6–7: literature survey and rationale for TF-IDF, Logistic Regression, median/MAD anomaly scoring, and moving averages.
- Pages 8–9: functional/non-functional requirements, architecture, data flow, and use cases.
- Pages 10–11: exact parsing, text-cleaning, classification, anomaly, duplicate, budget, and forecast methods.
- Pages 12–13: technology stack, module responsibilities, UI tabs, pseudocode, error handling, and original run instructions.
- Pages 14–16: testing strategy, results, functional test cases, and defects corrected.
- Pages 17–18: privacy, ethics, limitations, conclusion, and future work.
- Pages 20–24: appendix source code for the original prototype.
Where this guide recommends splitting the original single-file prototype into modules, that is a maintainability recommendation; the algorithms and expected behaviour remain based on the report.