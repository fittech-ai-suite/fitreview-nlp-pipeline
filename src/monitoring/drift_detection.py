"""
Data and model drift detection for the FitReview sentiment pipeline.

Run against a new batch of reviews to detect distribution shifts vs the
training reference set. Outputs an HTML report and a JSON summary.

Usage:
    python -m src.monitoring.drift_detection \
        --current data/current/new_reviews.csv \
        --reference data/processed/fitness_reviews_clean.csv \
        --report-dir results/drift
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def load_dataset(path: str, text_col: str = "review", label_col: str = "label") -> pd.DataFrame:
    df = pd.read_csv(path)
    available = df.columns.tolist()
    if text_col not in available:
        text_col = available[0]
    cols = [text_col]
    if label_col in available:
        cols.append(label_col)
    df = df[cols].dropna(subset=[text_col])
    df = df.rename(columns={text_col: "review"})
    if label_col in df.columns:
        df = df.rename(columns={label_col: "label"})
    df["review_length"] = df["review"].str.len()
    df["word_count"] = df["review"].str.split().str.len()
    return df


def run_drift_report(
    reference_path: str,
    current_path: str,
    report_dir: str = "results/drift",
) -> dict:
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.report import Report
    except ImportError:
        print(
            "evidently is not installed. Run: pip install evidently",
            file=sys.stderr,
        )
        sys.exit(1)

    Path(report_dir).mkdir(parents=True, exist_ok=True)

    reference = load_dataset(reference_path)
    current = load_dataset(current_path)

    # Only compare columns present in both datasets
    shared_cols = list(set(reference.columns) & set(current.columns))
    reference = reference[shared_cols]
    current = current[shared_cols]

    text_features = ["review_length", "word_count"]
    cat_features = ["label"] if "label" in shared_cols else []

    column_mapping = ColumnMapping(
        numerical_features=[c for c in text_features if c in shared_cols],
        categorical_features=cat_features,
        text_features=["review"] if "review" in shared_cols else [],
    )

    report = Report(metrics=[DataQualityPreset(), DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

    html_path = os.path.join(report_dir, "drift_report.html")
    report.save_html(html_path)
    print(f"HTML report saved: {html_path}")

    result = report.as_dict()
    drift_detected = any(
        m.get("result", {}).get("drift_detected", False)
        for m in result.get("metrics", [])
        if "drift_detected" in m.get("result", {})
    )

    summary = {
        "reference_rows": len(reference),
        "current_rows": len(current),
        "drift_detected": drift_detected,
        "columns_checked": shared_cols,
        "report_path": html_path,
    }

    json_path = os.path.join(report_dir, "drift_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {json_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Detect drift in review data")
    parser.add_argument("--reference", default="data/processed/fitness_reviews_clean.csv")
    parser.add_argument("--current", required=True, help="Path to new review CSV")
    parser.add_argument("--report-dir", default="results/drift")
    args = parser.parse_args()

    summary = run_drift_report(
        reference_path=args.reference,
        current_path=args.current,
        report_dir=args.report_dir,
    )

    drift_status = "DRIFT DETECTED" if summary["drift_detected"] else "No drift detected"
    print(f"\n{drift_status}")
    print(f"  Reference: {summary['reference_rows']} rows")
    print(f"  Current:   {summary['current_rows']} rows")

    if summary["drift_detected"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
