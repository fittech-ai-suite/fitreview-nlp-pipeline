"""Confusion matrix, calibration, and error analysis for the binary model."""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.inference.predictor import SentimentPredictor

CLEAN_CSV = "data/processed/fitness_reviews_clean.csv"
OUT_DIR = "results/evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_NAMES = ["negative", "positive"]


def load_test_split():
    df = pd.read_csv(CLEAN_CSV)
    df = df[df["label"] != 1].copy()
    df["label"] = df["label"].map({0: 0, 2: 1})
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    _, X_test, _, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    return X_test, y_test


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(LABEL_NAMES, fontsize=12)
    ax.set_yticklabels(LABEL_NAMES, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title("Confusion Matrix — Binary Sentiment Model", fontsize=13, pad=12)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return cm


def plot_calibration(y_true, predictions):
    confs = np.array([p["confidence"] for p in predictions])
    correct = np.array([
        int(p["label_id"] == t) for p, t in zip(predictions, y_true)
    ])

    bins = np.linspace(0.5, 1.0, 11)
    bin_acc, bin_conf, bin_count = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        bin_acc.append(correct[mask].mean())
        bin_conf.append(confs[mask].mean())
        bin_count.append(mask.sum())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=1, label="Perfect calibration")
    scatter = ax.scatter(bin_conf, bin_acc, c=bin_count, cmap="YlOrRd",
                         s=120, zorder=5, edgecolors="black", linewidths=0.5)
    plt.colorbar(scatter, ax=ax, label="Sample count")
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction Correct", fontsize=12)
    ax.set_title("Calibration Plot — Binary Sentiment Model", fontsize=13)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(0.48, 1.02)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "calibration.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_confidence_distribution(y_true, predictions):
    correct_confs = [p["confidence"] for p, t in zip(predictions, y_true) if p["label_id"] == t]
    wrong_confs = [p["confidence"] for p, t in zip(predictions, y_true) if p["label_id"] != t]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(correct_confs, bins=40, alpha=0.65, color="#2ecc71", label=f"Correct (n={len(correct_confs)})")
    ax.hist(wrong_confs, bins=40, alpha=0.65, color="#e74c3c", label=f"Wrong (n={len(wrong_confs)})")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Model Confidence", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Confidence Distribution: Correct vs Wrong Predictions", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confidence_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def error_analysis(texts, y_true, predictions, n=15):
    errors = [
        {
            "text": texts[i][:300],
            "true_label": LABEL_NAMES[y_true[i]],
            "predicted_label": predictions[i]["label"],
            "confidence": round(predictions[i]["confidence"], 4),
        }
        for i in range(len(texts))
        if predictions[i]["label_id"] != y_true[i]
    ]
    errors.sort(key=lambda x: x["confidence"], reverse=True)

    fp = [e for e in errors if e["predicted_label"] == "positive"][:n]
    fn = [e for e in errors if e["predicted_label"] == "negative"][:n]

    print(f"\n{'='*60}")
    print(f"  ERROR ANALYSIS  (total errors: {len(errors)} / {len(texts)})")
    print(f"{'='*60}")
    print(f"\n--- Top {len(fp)} False Positives (predicted +ve, actually -ve) ---")
    for i, e in enumerate(fp[:5], 1):
        print(f"\n  [{i}] conf={e['confidence']:.3f}")
        print(f"  \"{e['text'][:200]}\"")

    print(f"\n--- Top {len(fn)} False Negatives (predicted -ve, actually +ve) ---")
    for i, e in enumerate(fn[:5], 1):
        print(f"\n  [{i}] conf={e['confidence']:.3f}")
        print(f"  \"{e['text'][:200]}\"")

    out = {"total_errors": len(errors), "false_positives": fp, "false_negatives": fn}
    path = os.path.join(OUT_DIR, "error_analysis.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")
    return out


if __name__ == "__main__":
    print("Loading test split...")
    X_test, y_test = load_test_split()
    print(f"Test set: {len(X_test)} samples | neg: {y_test.count(0)} | pos: {y_test.count(1)}")

    print("\nLoading model and running inference...")
    predictor = SentimentPredictor()
    predictions = predictor.predict_batch(X_test, batch_size=64)

    y_pred = [p["label_id"] for p in predictions]

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    f1_per = f1_score(y_test, y_pred, average=None, labels=[0, 1])
    report = classification_report(y_test, y_pred, target_names=LABEL_NAMES, digits=3)

    print(f"\nAccuracy: {acc:.4f}  Macro-F1: {macro_f1:.4f}")
    print(f"F1 neg: {f1_per[0]:.3f}  F1 pos: {f1_per[1]:.3f}")
    print(f"\n{report}")

    print("Generating plots...")
    cm = plot_confusion_matrix(y_test, y_pred)
    plot_confidence_distribution(y_test, predictions)
    plot_calibration(y_test, predictions)

    print("\nRunning error analysis...")
    error_analysis(X_test, y_test, predictions)

    summary = {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "f1_negative": round(float(f1_per[0]), 3),
        "f1_positive": round(float(f1_per[1]), 3),
        "test_size": len(X_test),
        "confusion_matrix": cm.tolist(),
    }
    path = os.path.join(OUT_DIR, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {path}")
    print("\nDone.")
