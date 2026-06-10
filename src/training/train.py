import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import pipeline
import os
import json
import pickle

os.makedirs("models", exist_ok=True)

def load_data():
    df = pd.read_csv("data/raw/fitness_reviews.csv")
    print(f"Loaded {len(df)} reviews")
    print(df["sentiment"].value_counts())
    return df

def evaluate_model(df):
    print("\nLoading DistilBERT sentiment pipeline...")
    classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )

    print("Running predictions...")
    results = []
    for text in df["text"].tolist():
        result = classifier(text)[0]
        label = result["label"]
        score = result["score"]
        if label == "POSITIVE":
            pred = "positive"
        else:
            pred = "negative"
        results.append({"pred_sentiment": pred, "confidence": score})

    df_results = df.copy()
    df_results["pred_sentiment"] = [r["pred_sentiment"] for r in results]
    df_results["confidence"] = [r["confidence"] for r in results]

    binary_df = df_results[df_results["sentiment"].isin(["positive", "negative"])].copy()
    acc = accuracy_score(binary_df["sentiment"], binary_df["pred_sentiment"])
    f1 = f1_score(binary_df["sentiment"], binary_df["pred_sentiment"], average="weighted")

    print(f"\nBinary Accuracy (pos/neg): {acc:.4f}")
    print(f"F1 Weighted: {f1:.4f}")
    print("\nDetailed Report:")
    print(classification_report(binary_df["sentiment"], binary_df["pred_sentiment"]))

    return classifier, acc, f1

def save_model_info():
    model_info = {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "model_type": "DistilBERT",
        "task": "sentiment-analysis",
        "labels": ["negative", "neutral", "positive"],
        "version": "1.0.0"
    }
    with open("models/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    print("\nModel info saved to models/model_info.json")

if __name__ == "__main__":
    mlflow.set_experiment("fitreview-sentiment")

    with mlflow.start_run():
        df = load_data()
        classifier, acc, f1 = evaluate_model(df)

        mlflow.log_param("model", "distilbert-base-uncased-finetuned-sst-2-english")
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_metric("binary_accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        save_model_info()
        print("\nMLflow run complete!")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1: {f1:.4f}")
