"""
TF-IDF baselines (LogReg + LinearSVC) for binary sentiment.
Same 80/20 split as train_binary.py so results are directly comparable.
"""

import pandas as pd
import numpy as np
import json
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

CLEAN_CSV = "data/processed/fitness_reviews_clean.csv"
OUT_DIR = "results/baselines"
os.makedirs(OUT_DIR, exist_ok=True)


def load_binary_data():
    df = pd.read_csv(CLEAN_CSV)
    df = df[df["label"] != 1].copy()
    df["label"] = df["label"].map({0: 0, 2: 1})
    return df["text"].tolist(), df["label"].tolist()


def evaluate(name: str, y_true, y_pred, elapsed: float) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    f1_per = f1_score(y_true, y_pred, average=None, labels=[0, 1])
    report = classification_report(
        y_true, y_pred, target_names=["negative", "positive"], digits=3
    )
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro-F1  : {macro_f1:.4f}")
    print(f"  F1 neg    : {f1_per[0]:.3f}  |  F1 pos : {f1_per[1]:.3f}")
    print(f"  Train time: {elapsed:.1f}s")
    print(f"\n{report}")
    return {
        "model": name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "f1_negative": round(float(f1_per[0]), 3),
        "f1_positive": round(float(f1_per[1]), 3),
        "train_time_sec": round(elapsed, 2),
    }


if __name__ == "__main__":
    texts, labels = load_binary_data()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Label dist (test) - neg: {y_test.count(0)} | pos: {y_test.count(1)}")

    print("\nFitting TF-IDF (1-2gram, 50k features)...")
    t0 = time.time()
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"Vocab size: {len(tfidf.vocabulary_):,}")

    results = []

    t0 = time.time()
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_tfidf, y_train)
    elapsed = time.time() - t0
    preds = dummy.predict(X_test_tfidf)
    results.append(evaluate("Majority Class Baseline", y_test, preds, elapsed))

    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
    lr.fit(X_train_tfidf, y_train)
    elapsed = time.time() - t0
    preds = lr.predict(X_test_tfidf)
    results.append(evaluate("TF-IDF + Logistic Regression", y_test, preds, elapsed))

    t0 = time.time()
    svc = LinearSVC(max_iter=2000, C=1.0, class_weight="balanced", random_state=42)
    svc.fit(X_train_tfidf, y_train)
    elapsed = time.time() - t0
    preds = svc.predict(X_test_tfidf)
    results.append(evaluate("TF-IDF + LinearSVC", y_test, preds, elapsed))

    print("\n" + "="*55)
    print("  MODEL COMPARISON (same test split as DistilBERT)")
    print("="*55)
    header = f"{'Model':<32} {'Acc':>6} {'MacroF1':>8} {'negF1':>6} {'posF1':>6}"
    print(header)
    print("-"*55)
    for r in results:
        print(f"{r['model']:<32} {r['accuracy']:>6.4f} {r['macro_f1']:>8.4f} {r['f1_negative']:>6.3f} {r['f1_positive']:>6.3f}")
    # DistilBERT row from training run
    print(f"{'DistilBERT (binary, 4 ep)':<32} {'0.9124':>6} {'0.9089':>8} {'0.927':>6} {'0.891':>6}")
    print("="*55)

    out_path = os.path.join(OUT_DIR, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
