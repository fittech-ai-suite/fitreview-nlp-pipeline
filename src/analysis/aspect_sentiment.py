"""
Aggregate binary model predictions by app, category, and aspect to surface
what drives positive vs negative reviews across the fitness app landscape.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.inference.predictor import SentimentPredictor

CLEAN_CSV = "data/processed/fitness_reviews_clean.csv"
OUT_DIR = "results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

ASPECTS = {
    "asp_price": "Price / Value",
    "asp_bugs_crashes": "Bugs & Crashes",
    "asp_ads": "Ads",
    "asp_ui_ux": "UI / UX",
    "asp_features": "Features",
    "asp_accuracy": "Accuracy",
    "asp_support": "Support",
}


def load_data():
    df = pd.read_csv(CLEAN_CSV)
    df = df[df["label"] != 1].copy()
    df["label"] = df["label"].map({0: 0, 2: 1})
    return df


def plot_app_sentiment(df: pd.DataFrame):
    app_stats = (
        df.groupby("app")["pred_positive"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "pct_positive", "count": "n_reviews"})
        .sort_values("pct_positive", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, max(6, len(app_stats) * 0.45)))
    cmap = plt.cm.RdYlGn
    colors = [cmap(v) for v in app_stats["pct_positive"]]
    bars = ax.barh(app_stats.index, app_stats["pct_positive"] * 100, color=colors,
                   edgecolor="white", linewidth=0.5)

    for bar, (_, row) in zip(bars, app_stats.iterrows()):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{row['pct_positive']*100:.1f}%  (n={int(row['n_reviews'])})",
            va="center", fontsize=8.5, color="#333333",
        )

    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("% Positive Reviews", fontsize=12)
    ax.set_title("Sentiment by App — Binary DistilBERT Predictions", fontsize=13, pad=12)
    ax.set_xlim(0, 115)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "sentiment_by_app.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return app_stats


def plot_category_sentiment(df: pd.DataFrame):
    cat_stats = (
        df.groupby("category")["pred_positive"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "pct_positive", "count": "n_reviews"})
        .sort_values("pct_positive", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, max(4, len(cat_stats) * 0.5)))
    cmap = plt.cm.RdYlGn
    colors = [cmap(v) for v in cat_stats["pct_positive"]]
    bars = ax.barh(cat_stats.index, cat_stats["pct_positive"] * 100, color=colors,
                   edgecolor="white")
    for bar, (_, row) in zip(bars, cat_stats.iterrows()):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{row['pct_positive']*100:.1f}%  (n={int(row['n_reviews'])})",
            va="center", fontsize=9,
        )

    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("% Positive Reviews", fontsize=12)
    ax.set_title("Sentiment by Category", fontsize=13)
    ax.set_xlim(0, 115)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "sentiment_by_category.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return cat_stats


def plot_aspect_heatmap(df: pd.DataFrame):
    top_apps = (
        df.groupby("app").size()
        .loc[lambda s: s >= 30]
        .sort_values(ascending=False)
        .index.tolist()
    )
    df_top = df[df["app"].isin(top_apps)]

    matrix = pd.DataFrame(index=top_apps, columns=list(ASPECTS.keys()), dtype=float)
    counts = pd.DataFrame(index=top_apps, columns=list(ASPECTS.keys()), dtype=int)
    for col in ASPECTS:
        sub = df_top[df_top[col] == 1]
        grp = sub.groupby("app")["pred_positive"].agg(["mean", "count"])
        for app in top_apps:
            if app in grp.index and grp.loc[app, "count"] >= 5:
                matrix.loc[app, col] = grp.loc[app, "mean"]
                counts.loc[app, col] = int(grp.loc[app, "count"])
            else:
                matrix.loc[app, col] = np.nan
                counts.loc[app, col] = 0

    matrix = matrix.astype(float)

    fig, ax = plt.subplots(figsize=(13, max(7, len(top_apps) * 0.5)))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#e74c3c", "#f39c12", "#2ecc71"]
    )
    im = ax.imshow(matrix.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(ASPECTS)))
    ax.set_xticklabels(list(ASPECTS.values()), rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(top_apps)))
    ax.set_yticklabels(top_apps, fontsize=9)

    for i, app in enumerate(top_apps):
        for j, col in enumerate(ASPECTS.keys()):
            val = matrix.loc[app, col]
            n = counts.loc[app, col]
            if not np.isnan(val):
                txt_color = "white" if val < 0.35 or val > 0.75 else "black"
                ax.text(j, i, f"{val*100:.0f}%\n(n={n})",
                        ha="center", va="center", fontsize=7.5,
                        color=txt_color, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#aaaaaa")

    plt.colorbar(im, ax=ax, label="% Positive", fraction=0.025, pad=0.02)
    ax.set_title(
        "Aspect-Level Sentiment Heatmap\n(% positive among reviews mentioning each aspect)",
        fontsize=13, pad=12,
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "aspect_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_overall_aspects(df: pd.DataFrame):
    rows = []
    for col, label in ASPECTS.items():
        sub = df[df[col] == 1]
        if len(sub) < 10:
            continue
        pct = sub["pred_positive"].mean()
        rows.append({"aspect": label, "pct_positive": pct, "n": len(sub)})

    asp_df = pd.DataFrame(rows).sort_values("pct_positive")

    cmap = plt.cm.RdYlGn
    colors = [cmap(v) for v in asp_df["pct_positive"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(asp_df["aspect"], asp_df["pct_positive"] * 100, color=colors,
                   edgecolor="white")
    for bar, row in zip(bars, asp_df.itertuples()):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{row.pct_positive*100:.1f}%  (n={row.n})",
            va="center", fontsize=10,
        )
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("% Positive Reviews", fontsize=12)
    ax.set_title("Sentiment Breakdown by Review Aspect", fontsize=13)
    ax.set_xlim(0, 115)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "sentiment_by_aspect.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return asp_df


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"Scoring {len(df)} reviews (neg={int((df['label']==0).sum())} | pos={int((df['label']==1).sum())})")

    print("\nRunning inference on full dataset...")
    predictor = SentimentPredictor()
    preds = predictor.predict_batch(df["text"].tolist(), batch_size=64)

    df["pred_label_id"] = [p["label_id"] for p in preds]
    df["pred_label"] = [p["label"] for p in preds]
    df["pred_confidence"] = [p["confidence"] for p in preds]
    df["pred_positive"] = df["pred_label_id"].astype(float)

    overall_acc = (df["pred_label_id"] == df["label"]).mean()
    print(f"\nOverall accuracy on full dataset: {overall_acc:.4f}")

    print("\nGenerating visualisations...")
    app_stats = plot_app_sentiment(df)
    cat_stats = plot_category_sentiment(df)
    plot_overall_aspects(df)
    plot_aspect_heatmap(df)

    print("\n" + "="*60)
    print("  KEY INSIGHTS")
    print("="*60)

    print("\n--- Top 5 most positive apps ---")
    for app, row in app_stats.sort_values("pct_positive", ascending=False).head(5).iterrows():
        print(f"  {app:<30} {row['pct_positive']*100:.1f}%  (n={int(row['n_reviews'])})")

    print("\n--- Top 5 most negative apps ---")
    for app, row in app_stats.sort_values("pct_positive").head(5).iterrows():
        print(f"  {app:<30} {row['pct_positive']*100:.1f}%  (n={int(row['n_reviews'])})")

    print("\n--- Aspect pain points (sorted worst first) ---")
    for col, label in ASPECTS.items():
        sub = df[df[col] == 1]
        if len(sub) < 10:
            continue
        pct = sub["pred_positive"].mean()
        print(f"  {label:<22} {pct*100:.1f}% positive  (n={len(sub)})")

    summary = {
        "total_reviews": len(df),
        "overall_accuracy_on_full_dataset": round(overall_acc, 4),
        "by_app": app_stats.reset_index().rename(
            columns={"pct_positive": "pct_positive", "n_reviews": "n_reviews"}
        ).to_dict(orient="records"),
        "by_category": cat_stats.reset_index().to_dict(orient="records"),
        "by_aspect": {
            ASPECTS[col]: {
                "pct_positive": round(df[df[col] == 1]["pred_positive"].mean(), 3),
                "n_reviews": int((df[col] == 1).sum()),
            }
            for col in ASPECTS if (df[col] == 1).sum() >= 10
        },
    }
    path = os.path.join(OUT_DIR, "aspect_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {path}")
    print("\nDone.")
