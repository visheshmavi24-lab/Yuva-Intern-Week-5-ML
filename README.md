# YuvaIntern Week 5 - Machine Learning Algorithms and Model Evaluation

**Project:** Apple historical stock-price forecasting  
**Author:** Vishesh Mavi  
**Primary input:** `aapl_ticker.csv`  
**Target:** next available trading session's adjusted closing price (`Adj Close`), falling back to `Close`.

## Project alignment

The Week 4 repository describes a stock-price time-series analysis and expects a `Date` plus a closing-price column. The Week 6 repository identifies the capstone dataset as `aapl_ticker.csv` and an Apple historical-stock-price project. This Week 5 implementation follows that stock time-series direction and uses the Apple CSV named in Week 6 and the existing Week 5 repository.

The public Week 1-3 repository descriptions do not consistently identify the same dataset: Week 1 refers to Blinkit/grocery data, while Week 3 calls the data BigMart/Grocery Sales. Confirm with the internship mentor which dataset should be treated as the official continuous dataset if they expect strict Week 1-6 dataset continuity. Do not combine grocery-sales data and Apple stock data in one model.

## Files

- `task5_ml.py` - full reproducible ML pipeline, tuning, charts, metrics and results-based DOCX report generation.
- `requirements.txt` - pinned Python dependencies.
- `README.md` - project details and execution instructions.
- `Week5_ML_Model_Evaluation_Report.docx` - pre-run report structure and methodology. The script generates the completed, results-based report in `outputs/` after execution. Do not submit this pre-run copy in place of the generated report.

## Dataset setup

Place the exact historical CSV used for the stock-price analysis in this project folder and name it `aapl_ticker.csv`. It must contain:

- `Date`
- `Adj Close` (preferred) or `Close`
- Optional fields: `Open`, `High`, `Low`, `Volume`

The dataset is not included in this package. Use the same permitted source file already used in the internship work. Check its source and redistribution terms before adding it to a public repository. If the CSV is not permitted to be redistributed, keep it local and document its source in the report/README.

## Run on Windows PowerShell

```powershell
# Open PowerShell in the project folder
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python task5_ml.py --data aapl_ticker.csv
```

If activation is blocked, run without activating:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe task5_ml.py --data aapl_ticker.csv
```

If your CSV has a different name or is stored elsewhere:

```powershell
python task5_ml.py --data "C:\path\to\your\stock_data.csv"
```

## Methodology

1. Parse dates, sort chronologically, remove duplicate dates and invalid target rows.
2. Use current-session price and historical lag/rolling features to predict the next available trading session.
3. Keep the final chronological 20% as the hold-out test set; do not shuffle time-series rows.
4. Tune Ridge Regression, Random Forest and Gradient Boosting using `GridSearchCV` with `TimeSeriesSplit`.
5. Compare all models against a naive persistence baseline (next-session price = current-session price).
6. Report MAE, RMSE and R-squared on the hold-out test set. Select the best ML model by cross-validation RMSE, not by test-set tuning.
7. Generate actual-vs-predicted, residual, RMSE comparison and feature-contribution charts.
8. Generate a DOCX report containing metrics calculated from the real run.

## Outputs

After a successful run, `outputs/` contains:

- `model_metrics.csv`
- `test_predictions.csv`
- `run_summary.json`
- `best_model.joblib`
- `actual_vs_predicted.png`
- `residual_plot.png`
- `model_rmse_comparison.png`
- `feature_importance.png` (when supported by the selected model)
- `Week5_ML_Model_Evaluation_Report.docx` (final run-specific report)

## Submission checks

- Run the script successfully on the original permitted CSV.
- Open the generated DOCX and verify the actual metrics, test dates and charts.
- Verify that the metrics in the DOCX match `outputs/model_metrics.csv`.
- Create the repository with these four project files. After running successfully, add the generated `outputs/` evidence (CSV metrics, predictions, charts and completed DOCX) to the repository as the final evidence set. Do not upload the template in place of the run-generated report.
- Do not claim that the project has been evaluated or passed until YuvaIntern records the evaluation.

**Important:** Results are dataset- and period-specific. Do not copy ARIMA, Prophet or XGBoost scores from the Week 6 report into this Week 5 experiment; the models and evaluation procedure differ. Stock-price forecasts are uncertain and this project is not investment advice.
