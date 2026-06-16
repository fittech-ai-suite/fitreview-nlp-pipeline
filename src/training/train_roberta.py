"""
RoBERTa-base fine-tuning with negation-aware data augmentation and label smoothing.

Why each piece:
  - RoBERTa-base: 2× DistilBERT's capacity, better pre-training (dynamic masking,
    no NSP, larger batches) — fundamentally understands sentence structure better
  - Negation augmentation: the positional-bias bug is a data distribution problem;
    synthetic "positive opener, negative body" examples directly fix it
  - Label smoothing: prevents overconfident predictions on edge cases
  - Cosine LR with warmup: more stable convergence than linear decay
"""

import json
import logging
import os
import random

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── hyperparameters ───────────────────────────────────────────────────────────

BASE_MODEL = "roberta-base"
DATA_PATH = "data/processed/fitness_reviews_clean.csv"
OUTPUT_DIR = "models/roberta-fitness-binary"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.1
SEED = 42
LABELS = {0: "negative", 1: "positive"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── negation augmentation ─────────────────────────────────────────────────────

def negation_augmentation() -> list:
    """
    Hard contrastive examples that teach the model not to anchor on the opening clause.
    Covers three failure patterns: positive-opener/negative-body, resigned-tone negatives,
    and sarcasm-adjacent positives.
    """
    # — Pattern 1: Positive opener that flips negative —
    pos_openers = [
        "This app completely transformed my fitness journey",
        "I absolutely love this app and what it offers",
        "Hands down the best fitness tracker I've tried",
        "I've been using this for two years and genuinely loved it",
        "The UI is beautiful and the feature set looked impressive",
        "Outstanding results at first — really motivating",
        "Would have given five stars without hesitation",
        "My favourite fitness app by a mile",
        "Incredible tracking and a great experience initially",
        "This app genuinely changed how I train",
        "I recommended this to all my friends",
        "Amazing value for money when I first signed up",
        "The calorie tracking was accurate and the UI was clean",
        "I was a huge fan of this app",
        "This used to be the gold standard for fitness apps",
    ]
    neg_tails = [
        "but the latest update broke GPS and I lost three months of data. One star.",
        "until they locked every useful feature behind a £15/month paywall. Unacceptable.",
        "but it crashes every time I log a strength session now. Completely unusable.",
        "but customer support ignored my ticket for five weeks. Absolutely appalling.",
        "before they raised prices 300% without warning and gutted the free tier.",
        "but the calorie estimates are so wildly inaccurate they're actively harmful.",
        "until the October update, which introduced bugs that still haven't been fixed six months later.",
        "but it now drains my battery in two hours and uploads gigabytes of data in the background.",
        "but they've turned it into a bloated mess with features nobody asked for. I'm done.",
        "until I discovered the sleep tracking data is completely fabricated. Total scam.",
        "but there's no customer support and the app crashes constantly. Deleted.",
        "until the subscription price tripled overnight with no notice. Disgusting.",
        "but the recent ads make it unusable and the devs refuse to fix anything.",
        "but I lost all my data after an update and was told it was unrecoverable.",
        "but the new version removed offline mode and is useless without signal.",
    ]

    # — Pattern 2: Negative opener that flips positive —
    neg_openers = [
        "This app is an absolute disaster and I deeply regret buying it",
        "Worst fitness app on the market by a wide margin",
        "Complete waste of money — I demanded a full refund",
        "Stay away. This app deleted months of my workout history",
        "The developers should be embarrassed by the state of this product",
        "One star is too generous. Genuinely broken beyond repair",
        "Never been more disappointed in a subscription purchase",
        "Uninstalled after a week. Buggy, slow, and dishonest about pricing",
        "I gave this app a terrible review six months ago",
        "Had endless problems with this app from day one",
    ]
    pos_tails = [
        "but after the 5.0 update everything works flawlessly. Completely turned around. Five stars.",
        "until they fixed all the major bugs in December — now it's genuinely excellent.",
        "before the latest release sorted out every issue I had. Highly recommend now.",
        "until support reached out personally and resolved everything in 24 hours. Outstanding team.",
        "but I'm happy to say version 4.2 fixed all of it. Back to being my favourite app.",
        "but they've clearly listened to feedback and the new version is a massive improvement.",
        "until a recent update that fixed everything and added features I'd been wanting for years.",
    ]

    # — Pattern 3: Calm resigned tone (false negatives in original model) —
    calm_negatives = [
        ("Since the October update GPS tracking no longer works for outdoor runs. I've reported it twice. No response.", 0),
        ("The sync between my watch and the app stopped working three weeks ago. Still not fixed.", 0),
        ("I've been paying for premium for two years. The latest version removed half the features I paid for.", 0),
        ("The heart rate data hasn't been accurate since the firmware update. I've contacted support four times.", 0),
        ("Calorie counts are consistently 30-40% off according to my nutritionist. I've switched to another app.", 0),
        ("The app logs my sleep as 9 hours when I get 5. The data is meaningless.", 0),
        ("I've submitted three bug reports. The crashes continue. I understand software is hard but this is too much.", 0),
        ("Subscription auto-renewed at a price 40% higher than last year. No notification. No explanation.", 0),
    ]

    examples = []
    for opener in pos_openers:
        for tail in neg_tails:
            examples.append((f"{opener}, {tail}", 0))
    for opener in neg_openers:
        for tail in pos_tails:
            examples.append((f"{opener}, {tail}", 1))
    examples.extend(calm_negatives)
    return examples


# ── dataset ───────────────────────────────────────────────────────────────────

class ReviewDataset(Dataset):
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
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── loss ──────────────────────────────────────────────────────────────────────

class LabelSmoothingCE(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # soft targets: (1 - ε) on the correct class, ε/(n-1) on others
        smooth = torch.full_like(log_probs, self.smoothing / (n - 1))
        smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth * log_probs).sum(dim=-1)
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[targets]
        return loss.mean()


# ── training / eval ───────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(out.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds.extend(out.logits.argmax(-1).cpu().tolist())
            trues.extend(batch["labels"].tolist())
    return {
        "accuracy": accuracy_score(trues, preds),
        "macro_f1": f1_score(trues, preds, average="macro"),
        "per_class": f1_score(trues, preds, average=None, labels=[0, 1]).tolist(),
        "report": classification_report(trues, preds, target_names=["negative", "positive"], digits=3),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    device = get_device()
    log.info(f"device: {device}  |  base model: {BASE_MODEL}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # load + filter data
    df = pd.read_csv(DATA_PATH)
    df = df[df["label"] != 1].copy()
    df["label"] = df["label"].map({0: 0, 2: 1})
    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    log.info(f"real reviews: {len(df)}")

    # inject negation augmentation
    aug = negation_augmentation()
    aug_df = pd.DataFrame(aug, columns=["text", "label"])
    df = pd.concat([df, aug_df], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    log.info(f"after negation augmentation: {len(df)} total  ({len(aug)} synthetic)")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
    log.info(f"train: {len(train_df)}  |  test: {len(test_df)}")

    y_train = train_df["label"].values
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weights = torch.tensor(cw, dtype=torch.float32)
    log.info(f"class weights  neg={cw[0]:.3f}  pos={cw[1]:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2).to(device)

    train_ds = ReviewDataset(train_df["text"].tolist(), y_train.tolist(), tokenizer, MAX_LEN)
    test_ds = ReviewDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    criterion = LabelSmoothingCE(smoothing=LABEL_SMOOTHING, weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    mlflow.set_experiment("fitreview-roberta-binary")
    with mlflow.start_run(run_name="roberta-fgsm-adv"):
        mlflow.log_params({
            "base_model": BASE_MODEL,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "max_len": MAX_LEN,
            "label_smoothing": LABEL_SMOOTHING,
            "warmup_ratio": WARMUP_RATIO,
            "aug_examples": len(aug),
        })

        best_f1, best_epoch = 0.0, 0
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            metrics = evaluate(model, test_loader, device)

            log.info(
                f"epoch {epoch}/{EPOCHS}  loss={train_loss:.4f}  "
                f"acc={metrics['accuracy']:.4f}  macro-F1={metrics['macro_f1']:.4f}  "
                f"neg-F1={metrics['per_class'][0]:.3f}  pos-F1={metrics['per_class'][1]:.3f}"
            )
            log.info(f"\n{metrics['report']}")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "f1_negative": metrics["per_class"][0],
                "f1_positive": metrics["per_class"][1],
            }, step=epoch)

            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                best_epoch = epoch
                model.save_pretrained(OUTPUT_DIR)
                tokenizer.save_pretrained(OUTPUT_DIR)
                log.info(f"  → new best saved  (macro-F1 {best_f1:.4f})")

        log.info(f"\nbest epoch: {best_epoch}  |  best macro-F1: {best_f1:.4f}")
        mlflow.log_metric("best_macro_f1", best_f1)

        info = {
            "base_model": BASE_MODEL,
            "best_epoch": best_epoch,
            "best_macro_f1": round(best_f1, 4),
            "label_map": LABELS,
            "training": {
                "label_smoothing": LABEL_SMOOTHING,
                "negation_augmentation": len(aug),
                "cosine_lr_schedule": True,
                "warmup_ratio": WARMUP_RATIO,
            },
        }
        with open(os.path.join(OUTPUT_DIR, "model_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        log.info(f"saved to {OUTPUT_DIR}")
        log.info("=" * 60)
        log.info(f"final  acc={metrics['accuracy']:.4f}  macro-F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
