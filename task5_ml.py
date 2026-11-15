"""\
YuvaIntern Week 5: Machine Learning Algorithms and Model Evaluation

Forecast the next trading day's Apple share closing price from historical
market information. The script performs a chronological hold-out evaluation,
time-series cross-validation, hyperparameter tuning, model comparison,
visualization, and generation of a results-based DOCX report.

Expected input: CSV with Date and either Adj Close or Close.
Preferred filename: aapl_ticker.csv

No performance figures are hard-coded. All reported metrics are calculated
from the CSV supplied at run time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs"
REPORT_NAME = "Week5_ML_Model_Evaluation_Report.docx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the YuvaIntern Week 5 machine-learning workflow."
    )
    parser.add_argument(
        "--data", default="aapl_ticker.csv",
        help="Input CSV path (default: aapl_ticker.csv in this project folder)."
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT),
        help="Directory for metrics, predictions, charts, model and DOCX report."
    )
    parser.add_argument(
        "--test-size", type=float, default=0.20,
        help="Chronological test fraction (default: 0.20)."
    )
    parser.add_argument(
        "--cv-splits", type=int, default=5,
        help="Maximum number of TimeSeriesSplit folds (default: 5)."
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random state for reproducible tree models (default: 42)."
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_and_prepare(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    """Load CSV and build leakage-aware one-step-ahead features."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Place aapl_ticker.csv in this project folder, or run with "
            "--data \"C:\\path\\to\\your\\stock_data.csv\"."
        )

    raw = pd.read_csv(csv_path)
    raw.columns = [str(c).strip() for c in raw.columns]
    original_rows = int(len(raw))

    # Match expected column names without changing the underlying source data.
    lower_to_actual = {str(c).strip().lower(): c for c in raw.columns}
    if "date" not in lower_to_actual:
        raise ValueError("The CSV must contain a Date column.")
    date_col = lower_to_actual["date"]
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")

    adj_col = lower_to_actual.get("adj close") or lower_to_actual.get("adj_close")
    close_col = lower_to_actual.get("close")
    if adj_col is not None:
        target_col = adj_col
        target_label = "Adjusted Close"
    elif close_col is not None:
        target_col = close_col
        target_label = "Close"
    else:
        raise ValueError("The CSV must contain either Adj Close or Close.")

    # Normalize known market-data headers while keeping the original file intact.
    rename_map: dict[str, str] = {date_col: "Date", target_col: "TargetPrice"}
    for expected in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        actual = lower_to_actual.get(expected.lower())
        if actual is not None and actual not in rename_map:
            rename_map[actual] = expected
    df = raw.rename(columns=rename_map).copy()

    # Parse all available numeric price/volume fields defensively.
    numeric_candidates = [
        c for c in ["TargetPrice", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if c in df.columns
    ]
    for col in numeric_candidates:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid_date_rows = int(df["Date"].isna().sum())
    df = df.dropna(subset=["Date"]).copy()
    duplicate_dates = int(df.duplicated(subset=["Date"]).sum())
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    missing_target_rows = int(df["TargetPrice"].isna().sum())
    df = df.dropna(subset=["TargetPrice"]).copy().reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(
            f"Only {len(df)} valid dated price rows remain. At least 100 are "
            "recommended for lag features, time-series cross-validation and a hold-out test."
        )

    # Current-session data is used to predict the NEXT available trading session.
    # For example, price_lag_1 is today's observed price and the target is the next row's price.
    features = pd.DataFrame(index=df.index)
    features["price_lag_1"] = df["TargetPrice"]
    for lag in [2, 3, 5, 10]:
        features[f"price_lag_{lag}"] = df["TargetPrice"].shift(lag - 1)

    daily_return = df["TargetPrice"].pct_change()
    features["return_1d"] = daily_return
    features["return_5d"] = df["TargetPrice"].pct_change(5)
    for window in [5, 10, 20]:
        features[f"rolling_mean_{window}"] = df["TargetPrice"].rolling(window).mean()
    for window in [5, 10, 20]:
        features[f"rolling_std_{window}"] = df["TargetPrice"].rolling(window).std()

    # OHLCV values from the completed current session are available for a next-session forecast.
    # Exclude the unadjusted Close field when Adjusted Close is the target (and vice versa).
    target_is_adjusted = target_label == "Adjusted Close"
    for col in ["Open", "High", "Low", "Volume"]:
        if col in df.columns:
            features[f"session_{col.lower()}"] = df[col]
    if not target_is_adjusted and "Adj Close" in df.columns:
        features["session_adj_close"] = df["Adj Close"]
    if target_is_adjusted and "Close" in df.columns:
        # Avoid mixing the unadjusted close with an adjusted-price target.
        pass

    # Calendar information from the current observed session.
    features["day_of_week"] = df["Date"].dt.dayofweek
    features["month"] = df["Date"].dt.month

    target = df["TargetPrice"].shift(-1).rename("ActualNextPrice")
    target_dates = df["Date"].shift(-1).rename("TargetDate")

    model_frame = features.replace([np.inf, -np.inf], np.nan)
    valid = target.notna() & target_dates.notna()
    required_feature_cols = [c for c in model_frame.columns if not c.startswith("session_")]
    valid &= model_frame[required_feature_cols].notna().all(axis=1)
    X = model_frame.loc[valid].copy()
    y = target.loc[valid].astype(float).copy()
    dates = target_dates.loc[valid].copy()

    if len(X) < 80:
        raise ValueError(
            f"Only {len(X)} usable feature rows remain after lag/rolling calculations. "
            "At least 80 are required; a longer CSV history may be needed."
        )

    audit = {
        "source_file": csv_path.name,
        "original_rows": original_rows,
        "valid_price_rows": int(len(df)),
        "invalid_date_rows_removed": invalid_date_rows,
        "duplicate_dates_removed": duplicate_dates,
        "missing_target_rows_removed": missing_target_rows,
        "feature_rows_removed_for_lags_or_missing_values": int(len(df) - len(X)),
        "usable_rows": int(len(X)),
        "first_source_date": df["Date"].min().strftime("%Y-%m-%d"),
        "last_source_date": df["Date"].max().strftime("%Y-%m-%d"),
        "target_column": target_label,
        "feature_columns": list(X.columns),
    }
    return df, X, y, dates, audit


def root_mean_squared_error(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_models(random_state: int) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    return {
        "Ridge Regression": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        "Random Forest": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(random_state=random_state, n_jobs=-1)),
            ]),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 6, None],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        "Gradient Boosting": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(random_state=random_state)),
            ]),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }


def save_charts(
    output_dir: Path,
    dates: pd.Series,
    y_test: pd.Series,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    best_model_name: str,
    feature_names: list[str],
    best_estimator: Pipeline,
) -> list[Path]:
    chart_paths: list[Path] = []
    plt.rcParams.update({"figure.figsize": (10, 5.5), "axes.grid": True, "grid.alpha": 0.25})

    # Actual versus best model and persistence baseline.
    fig, ax = plt.subplots()
    ax.plot(dates, y_test.values, label="Actual next-day price", linewidth=2)
    ax.plot(dates, predictions[best_model_name].values, label=f"{best_model_name} prediction", linewidth=1.6)
    ax.plot(dates, predictions["Naive Baseline"].values, label="Naive previous-price baseline", linewidth=1.2, alpha=0.8)
    ax.set_title("Actual vs Predicted Next-Trading-Day Price")
    ax.set_xlabel("Target date")
    ax.set_ylabel("Price (dataset currency/unit)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    path = output_dir / "actual_vs_predicted.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_paths.append(path)

    # Residual diagnostics.
    residuals = y_test.values - predictions[best_model_name].values
    fig, ax = plt.subplots()
    ax.scatter(predictions[best_model_name].values, residuals, alpha=0.7)
    ax.axhline(0, linestyle="--", linewidth=1.2)
    ax.set_title(f"Residual Plot - {best_model_name}")
    ax.set_xlabel("Predicted price")
    ax.set_ylabel("Residual (actual - predicted)")
    fig.tight_layout()
    path = output_dir / "residual_plot.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_paths.append(path)

    # RMSE comparison.
    plot_metrics = metrics.sort_values("RMSE", ascending=True)
    fig, ax = plt.subplots()
    ax.bar(plot_metrics["Model"], plot_metrics["RMSE"])
    ax.set_title("Test RMSE Comparison (Lower Is Better)")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    path = output_dir / "model_rmse_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    chart_paths.append(path)

    # Feature importance / standardized coefficient magnitude for interpretation.
    fitted_model = best_estimator.named_steps["model"]
    importance: np.ndarray | None = None
    if hasattr(fitted_model, "feature_importances_"):
        importance = np.asarray(fitted_model.feature_importances_, dtype=float)
    elif hasattr(fitted_model, "coef_"):
        importance = np.abs(np.asarray(fitted_model.coef_, dtype=float).ravel())
    if importance is not None and len(importance) == len(feature_names):
        imp = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        imp = imp.sort_values("Importance", ascending=False).head(12)
        fig, ax = plt.subplots()
        ax.barh(imp["Feature"].iloc[::-1], imp["Importance"].iloc[::-1])
        ax.set_title(f"Top Feature Contributions - {best_model_name}")
        ax.set_xlabel("Importance" if hasattr(fitted_model, "feature_importances_") else "Absolute standardized coefficient")
        fig.tight_layout()
        path = output_dir / "feature_importance.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        chart_paths.append(path)

    return chart_paths


def set_doc_styles(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    styles = doc.styles
    styles["Normal"].font.name = "Aptos"
    styles["Normal"].font.size = Pt(9.5)
    styles["Normal"].paragraph_format.space_after = Pt(5)
    for style_name, size in [("Title", 25), ("Heading 1", 16), ("Heading 2", 12)]:
        styles[style_name].font.name = "Aptos Display"
        styles[style_name].font.size = Pt(size)
        styles[style_name].font.color.rgb = RGBColor(23, 54, 93)


def add_table(doc: Document, headers: list[str], rows: list[list[Any]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    repeat = __import__("docx").oxml.OxmlElement("w:tblHeader")
    repeat.set(__import__("docx").oxml.ns.qn("w:val"), "true")
    table.rows[0]._tr.get_or_add_trPr().append(repeat)
    for cell, text in zip(table.rows[0].cells, headers):
        cell.text = str(text)
    for row in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = str(text)


def add_picture_if_present(doc: Document, path: Path | None, caption: str) -> None:
    if path is not None and path.is_file():
        doc.add_picture(str(path), width=Inches(6.4))
        p = doc.add_paragraph(caption)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def generate_report(
    report_path: Path,
    csv_path: Path,
    audit: dict[str, Any],
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    cv_splits: int,
    split_date: str,
    best_model_name: str,
    best_params: dict[str, Any],
    all_best_params: dict[str, dict[str, Any]],
    chart_paths: list[Path],
    target_dates: pd.Series,
) -> None:
    doc = Document()
    set_doc_styles(doc)

    title = doc.add_paragraph()
    title.style = "Title"
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("Week 5: Machine Learning Algorithms\nand Model Evaluation")
    subtitle = doc.add_paragraph("Apple Historical Stock Price - Next-Trading-Day Regression")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].bold = True
    p = doc.add_paragraph(f"Prepared by: Vishesh Mavi | Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("1. Executive Summary", level=1)
    ml_rows = metrics[metrics["Model"] != "Naive Baseline"].sort_values("CV_RMSE")
    top_test = metrics.loc[metrics["Model"] == best_model_name].iloc[0]
    baseline = metrics.loc[metrics["Model"] == "Naive Baseline"].iloc[0]
    if top_test["RMSE"] < baseline["RMSE"]:
        comparison = f"{best_model_name} achieved a lower test RMSE than the naive baseline by {baseline['RMSE'] - top_test['RMSE']:.4f}."
    else:
        comparison = f"{best_model_name} did not outperform the naive baseline on test RMSE; its RMSE was {top_test['RMSE'] - baseline['RMSE']:.4f} higher."
    doc.add_paragraph(
        f"This project evaluates next-trading-day Apple share-price prediction using historical daily observations. "
        f"Three supervised regression algorithms were tuned using {cv_splits}-fold expanding-window time-series cross-validation. "
        f"The best model was selected using cross-validation RMSE, then evaluated once on a later chronological hold-out set. "
        f"{comparison} The experiment is an educational forecasting exercise and is not investment advice."
    )
    add_table(doc, ["Item", "Run-specific result"], [
        ["Input dataset", audit["source_file"]],
        ["Target", f"Next-session {audit['target_column']}"] ,
        ["Usable supervised observations", audit["usable_rows"]],
        ["Training observations", len(predictions.attrs.get("train_count", [])) if isinstance(predictions.attrs.get("train_count"), list) else predictions.attrs.get("train_count", "See split details")],
        ["Test period begins", split_date],
        ["Best model (CV RMSE selection)", best_model_name],
        ["Best model test MAE", f"{top_test['MAE']:.4f}"],
        ["Best model test RMSE", f"{top_test['RMSE']:.4f}"],
        ["Best model test R-squared", f"{top_test['R2']:.4f}"],
    ])

    doc.add_heading("2. Objective and Problem Definition", level=1)
    doc.add_paragraph(
        "The objective is to implement regression algorithms, compare them with a simple baseline, tune model parameters, "
        "and evaluate predictive performance using data that occurs after the training period. The target is the next available "
        "trading session's adjusted closing price when an Adj Close column is available; otherwise, the closing price is used."
    )
    doc.add_paragraph(
        "Problem statement: Given information available at the end of a trading session, estimate the Apple share price at the "
        "next available trading session. The experiment evaluates one-step-ahead forecasts in a rolling setting: each test row's "
        "features use only that row's current-session information and earlier observations."
    )

    doc.add_heading("3. Dataset and Data Quality Review", level=1)
    add_table(doc, ["Data audit item", "Observed value"], [
        ["Source file", audit["source_file"]],
        ["Original CSV rows", audit["original_rows"]],
        ["Valid dated price rows", audit["valid_price_rows"]],
        ["Invalid date rows removed", audit["invalid_date_rows_removed"]],
        ["Duplicate dates removed", audit["duplicate_dates_removed"]],
        ["Rows with missing target removed", audit["missing_target_rows_removed"]],
        ["Rows removed by lag/rolling feature preparation", audit["feature_rows_removed_for_lags_or_missing_values"]],
        ["Final usable observations", audit["usable_rows"]],
        ["Source date range", f"{audit['first_source_date']} to {audit['last_source_date']}"],
        ["Target variable", audit["target_column"]],
    ])
    doc.add_paragraph(
        "Dates were parsed, records were sorted in ascending order, duplicate dates were reduced to one observation, and rows "
        "without a valid target price were removed. Lagged and rolling features require an initial history window, so early rows "
        "without complete feature values are excluded. No target values are synthetically filled."
    )

    doc.add_heading("4. Feature Engineering and Preprocessing", level=1)
    doc.add_paragraph(
        "Features include the current and lagged target prices (1, 2, 3, 5 and 10 session offsets), one- and five-session returns, "
        "5/10/20-session rolling means and rolling standard deviations, available current-session Open/High/Low/Volume fields, "
        "and calendar indicators. These values are formed from the current session and its history to predict the next available "
        "trading session. Raw target-price columns are not inserted as contemporaneous independent variables."
    )
    doc.add_paragraph(
        "A median imputer is placed inside each model pipeline so that imputation is learned from training folds only. Ridge "
        "Regression additionally uses StandardScaler inside its pipeline. Tree-based models do not require feature scaling."
    )
    doc.add_paragraph("Selected model inputs: " + ", ".join(audit["feature_columns"]))

    doc.add_heading("5. Train/Test Strategy and Cross-Validation", level=1)
    doc.add_paragraph(
        f"The usable observations were kept in chronological order and divided into an 80% training segment and a 20% test segment. "
        f"The first target date in the hold-out test segment was {split_date}. Random shuffling was not used. Hyperparameter tuning "
        f"used TimeSeriesSplit with {cv_splits} expanding-window folds on the training segment. GridSearchCV selected each model's "
        f"configuration using negative root mean squared error (equivalent to minimizing RMSE)."
    )

    doc.add_heading("6. Algorithms and Hyperparameter Tuning", level=1)
    doc.add_paragraph(
        "Ridge Regression provides a regularized linear benchmark. Random Forest averages predictions from an ensemble of decision "
        "trees and can represent nonlinear relationships. Gradient Boosting builds an additive sequence of trees that focuses on "
        "remaining prediction errors. A naive persistence baseline predicts the next-session price using the current-session price."
    )
    add_table(doc, ["Algorithm", "Best parameters selected by CV"], [
        ["Naive Baseline", "No fitted parameters; persistence forecast"],
        ["Ridge Regression", json.dumps(all_best_params.get("Ridge Regression", {}), sort_keys=True)],
        ["Random Forest", json.dumps(all_best_params.get("Random Forest", {}), sort_keys=True)],
        ["Gradient Boosting", json.dumps(all_best_params.get("Gradient Boosting", {}), sort_keys=True)],
    ])
    doc.add_paragraph("Selected best-model parameters: " + json.dumps(best_params, default=str, sort_keys=True))

    doc.add_heading("7. Model Evaluation Results", level=1)
    doc.add_paragraph(
        "MAE measures average absolute prediction error in the target price's units. RMSE gives larger errors more weight and is "
        "used for tuning and model comparison. R-squared summarizes the proportion of test-set variance explained relative to a "
        "constant-mean reference; it can be negative on unseen data. Lower MAE/RMSE is better, while higher R-squared is better."
    )
    result_rows = []
    for _, row in metrics.sort_values("RMSE").iterrows():
        result_rows.append([
            row["Model"],
            f"{row['CV_RMSE']:.4f}" if pd.notna(row["CV_RMSE"]) else "N/A",
            f"{row['MAE']:.4f}",
            f"{row['RMSE']:.4f}",
            f"{row['R2']:.4f}",
        ])
    add_table(doc, ["Model", "CV RMSE", "Test MAE", "Test RMSE", "Test R-squared"], result_rows)
    doc.add_paragraph(
        f"The machine-learning model selected from cross-validation was {best_model_name}. Its test RMSE was "
        f"{top_test['RMSE']:.4f}, compared with {baseline['RMSE']:.4f} for the naive baseline. "
        "The final interpretation should consider both the absolute error and the baseline comparison; a complex model is not "
        "automatically useful if it does not improve on the simple baseline."
    )

    doc.add_heading("8. Visual Evaluation", level=1)
    path_map = {p.name: p for p in chart_paths}
    add_picture_if_present(doc, path_map.get("actual_vs_predicted.png"), "Figure 1. Actual next-session prices compared with the selected model and persistence baseline.")
    add_picture_if_present(doc, path_map.get("residual_plot.png"), "Figure 2. Residuals for the selected model; visible structure may indicate systematic errors.")
    add_picture_if_present(doc, path_map.get("model_rmse_comparison.png"), "Figure 3. Hold-out RMSE comparison across the baseline and tuned machine-learning models.")
    add_picture_if_present(doc, path_map.get("feature_importance.png"), "Figure 4. Model-specific feature contribution view. This is descriptive, not causal.")

    doc.add_heading("9. Interpretation, Limitations and Future Improvements", level=1)
    doc.add_paragraph(
        "Historical price features may help describe recent price level and momentum, but stock prices are affected by news, "
        "market conditions, macroeconomic events, corporate actions and other variables not represented in this dataset. A "
        "single historical hold-out period does not establish that performance will remain stable in other market regimes. "
        "Adjusted and unadjusted price series are not interchangeable; this run uses the adjusted-close field when present. "
        "The one-step-ahead evaluation assumes the latest completed session's values become available before the next session's forecast."
    )
    doc.add_paragraph(
        "Future work can add a longer and independently sourced dataset, rolling-origin evaluation across several test windows, "
        "transaction-cost-aware directional evaluation, prediction intervals, additional exogenous variables, and comparisons "
        "with domain-specific forecasting methods. Any such extension should preserve chronological validation and avoid using "
        "future information in feature construction."
    )

    doc.add_heading("10. Conclusion", level=1)
    doc.add_paragraph(
        f"The experiment completed the supervised regression workflow using {len(metrics) - 1} tuned machine-learning algorithms "
        f"and a persistence baseline. {best_model_name} was selected using cross-validation RMSE and then compared on the held-out "
        "period. The reported metrics and charts in this document were generated directly from the supplied CSV during this run. "
        "The findings describe historical model performance only and should not be interpreted as financial advice."
    )

    doc.add_heading("11. Reproducibility", level=1)
    doc.add_paragraph(
        "The analysis is implemented in task5_ml.py. Install the pinned project dependencies from requirements.txt and run "
        "the script with the original CSV. The script saves model_metrics.csv, test_predictions.csv, run_summary.json, "
        "best_model.joblib, charts and this report under the outputs directory."
    )

    # Header/footer for a neat multi-page submission document.
    for section in doc.sections:
        header = section.header.paragraphs[0]
        header.text = "YUVA INTERN | WEEK 5 | MACHINE LEARNING MODEL EVALUATION"
        header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        if header.runs:
            header.runs[0].font.size = Pt(8)
            header.runs[0].font.color.rgb = RGBColor(100, 100, 100)
        footer = section.footer.paragraphs[0]
        footer.text = "Vishesh Mavi | Week 5 Resubmission"
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if footer.runs:
            footer.runs[0].font.size = Pt(8)
            footer.runs[0].font.color.rgb = RGBColor(100, 100, 100)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(report_path)


def main() -> int:
    args = parse_args()
    if not (0.10 <= args.test_size <= 0.40):
        raise ValueError("--test-size must be between 0.10 and 0.40.")
    if args.cv_splits < 2:
        raise ValueError("--cv-splits must be at least 2.")

    csv_path = resolve_path(args.data)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw, X, y, target_dates, audit = load_and_prepare(csv_path)
    split_idx = int(math.floor(len(X) * (1.0 - args.test_size)))
    if split_idx < 50 or (len(X) - split_idx) < 10:
        raise ValueError("Not enough observations for the requested chronological train/test split.")

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_dates = target_dates.iloc[split_idx:]
    cv_n_splits = min(args.cv_splits, max(2, min(5, len(X_train) // 12)))
    cv = TimeSeriesSplit(n_splits=cv_n_splits)

    model_specs = build_models(args.random_state)
    rows: list[dict[str, Any]] = []
    test_predictions = pd.DataFrame({"Date": test_dates.dt.strftime("%Y-%m-%d").values, "Actual": y_test.values})
    # Persistence baseline: forecast next session with the current session's observed price.
    naive_pred = X_test["price_lag_1"].to_numpy(dtype=float)
    test_predictions["Naive Baseline"] = naive_pred
    rows.append({
        "Model": "Naive Baseline", "CV_RMSE": np.nan,
        "MAE": mean_absolute_error(y_test, naive_pred),
        "RMSE": root_mean_squared_error(y_test, naive_pred),
        "R2": r2_score(y_test, naive_pred),
        "Best_Parameters": "Persistence: next price = current price",
    })

    fitted_estimators: dict[str, Pipeline] = {}
    best_params_by_model: dict[str, dict[str, Any]] = {}

    print("=" * 72)
    print("WEEK 5 - MACHINE LEARNING MODEL EVALUATION")
    print("=" * 72)
    print(f"Dataset: {csv_path.name}")
    print(f"Target: next-session {audit['target_column']}")
    print(f"Usable observations: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Features: {len(X.columns)} | Time-series CV folds: {cv_n_splits}")

    for model_name, (pipeline, grid) in model_specs.items():
        print(f"\nTuning {model_name} ...")
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        estimator = search.best_estimator_
        pred = estimator.predict(X_test)
        fitted_estimators[model_name] = estimator
        best_params_by_model[model_name] = search.best_params_
        test_predictions[model_name] = pred
        rows.append({
            "Model": model_name,
            "CV_RMSE": -float(search.best_score_),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": root_mean_squared_error(y_test, pred),
            "R2": r2_score(y_test, pred),
            "Best_Parameters": json.dumps(search.best_params_, sort_keys=True),
        })
        print(f"  Best CV RMSE: {-search.best_score_:.4f}")
        print(f"  Test MAE: {rows[-1]['MAE']:.4f} | Test RMSE: {rows[-1]['RMSE']:.4f} | R2: {rows[-1]['R2']:.4f}")
        print(f"  Parameters: {search.best_params_}")

    metrics = pd.DataFrame(rows)
    ml_metrics = metrics[metrics["Model"] != "Naive Baseline"].sort_values("CV_RMSE")
    best_model_name = str(ml_metrics.iloc[0]["Model"])
    best_estimator = fitted_estimators[best_model_name]
    best_params = best_params_by_model[best_model_name]

    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    joblib.dump(best_estimator, output_dir / "best_model.joblib")

    chart_paths = save_charts(
        output_dir, test_dates, y_test, test_predictions, metrics,
        best_model_name, list(X.columns), best_estimator,
    )

    split_date = pd.Timestamp(test_dates.iloc[0]).strftime("%Y-%m-%d")
    # Include the counts explicitly in the report's dataset summary table.
    test_predictions.attrs["train_count"] = int(len(X_train))
    report_path = output_dir / REPORT_NAME
    generate_report(
        report_path=report_path,
        csv_path=csv_path,
        audit=audit,
        metrics=metrics,
        predictions=test_predictions,
        cv_splits=cv_n_splits,
        split_date=split_date,
        best_model_name=best_model_name,
        best_params=best_params,
        all_best_params=best_params_by_model,
        chart_paths=chart_paths,
        target_dates=test_dates,
    )

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(csv_path),
        "audit": audit,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_start_date": split_date,
        "cv_splits": int(cv_n_splits),
        "best_model_selected_by_cv_rmse": best_model_name,
        "best_parameters": best_params,
        "metrics": metrics.to_dict(orient="records"),
        "outputs": [p.name for p in sorted(output_dir.iterdir()) if p.is_file()],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 72)
    print("FINAL TEST METRICS")
    print(metrics[["Model", "CV_RMSE", "MAE", "RMSE", "R2"]].sort_values("RMSE").to_string(index=False))
    print(f"\nBest ML model selected by CV RMSE: {best_model_name}")
    print(f"Report: {report_path}")
    print(f"Outputs folder: {output_dir}")
    print("Review the generated report and charts before submitting.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
