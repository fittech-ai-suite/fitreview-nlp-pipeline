"""DistilBERT binary sentiment — negative vs positive, neutral reviews dropped."""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import mlflow
import os
import json
from tqdm import tqdm

os.makedirs("models/distilbert-fitness-binary", exist_ok=True)

LABELS = {0: "negative", 1: "positive"}
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CLEAN_CSV = "data/processed/fitness_reviews_clean.csv"


class FitnessReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_data():
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(f"{CLEAN_CSV} not found - run the EDA script first")
    df = pd.read_csv(CLEAN_CSV)
    print(f"Loaded {len(df)} reviews (3-class)")

    df = df[df["label"] != 1].copy()
    df["label"] = df["label"].map({0: 0, 2: 1})

    print(f"After dropping neutral: {len(df)} reviews")
    print(df["label"].value_counts().sort_index().rename(LABELS))
    return df


def compute_weights(labels_array):
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=labels_array)
    print(f"Class weights (neg, pos): {weights.tolist()}")
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)


def train_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_per = f1_score(all_labels, all_preds, average=None, labels=[0, 1])
    report = classification_report(
        all_labels, all_preds, target_names=list(LABELS.values()), digits=3
    )
    return acc, f1_macro, f1_per, report


if __name__ == "__main__":
    mlflow.set_experiment("fitreview-distilbert-binary")

    with mlflow.start_run():
        df = load_data()

        texts = df["text"].tolist()
        labels = df["label"].tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

        class_weights = compute_weights(np.array(y_train))

        print("\nLoading tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        train_ds = FitnessReviewDataset(X_train, y_train, tokenizer, MAX_LEN)
        test_ds = FitnessReviewDataset(X_test, y_test, tokenizer, MAX_LEN)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        print("Loading model...")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        ).to(DEVICE)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
        )

        mlflow.log_params({
            "model": "distilbert-base-uncased",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "max_len": MAX_LEN,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "device": str(DEVICE),
            "weighted_loss": True,
            "num_labels": 2,
            "data": "binary_no_neutral",
        })

        print(f"\nTraining for {EPOCHS} epochs...")
        best_f1 = 0
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, DEVICE)
            acc, f1_macro, f1_per, report = eval_epoch(model, test_loader, DEVICE)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Acc: {acc:.4f} | Macro-F1: {f1_macro:.4f}")
            print(f"Per-class F1 -> neg {f1_per[0]:.3f} | pos {f1_per[1]:.3f}")
            print(f"\n{report}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_acc": acc,
                "val_macro_f1": f1_macro,
                "val_f1_negative": float(f1_per[0]),
                "val_f1_positive": float(f1_per[1]),
            }, step=epoch)

            if f1_macro > best_f1:
                best_f1 = f1_macro
                model.save_pretrained("models/distilbert-fitness-binary")
                tokenizer.save_pretrained("models/distilbert-fitness-binary")
                print(f"New best model saved (macro-F1 {best_f1:.4f})")

        model_info = {
            "model_name": "distilbert-base-uncased-finetuned-fitness-binary",
            "base_model": "distilbert-base-uncased",
            "num_labels": 2,
            "labels": LABELS,
            "best_macro_f1": round(best_f1, 4),
            "weighted_loss": True,
            "version": "1.0.0",
        }
        with open("models/distilbert-fitness-binary/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        mlflow.log_metric("best_macro_f1", best_f1)

        print("\n" + "=" * 50)
        print(f"Done. Best macro-F1: {best_f1:.4f}")
        print("Saved to models/distilbert-fitness-binary/")
        print("=" * 50)
