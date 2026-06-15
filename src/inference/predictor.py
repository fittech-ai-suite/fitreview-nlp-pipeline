import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
from typing import List, Dict, Optional

LABELS = {0: "negative", 1: "positive"}


class SentimentPredictor:
    def __init__(
        self,
        model_dir: str = "models/distilbert-fitness-binary",
        device: Optional[str] = None,
        max_len: int = 128,
    ) -> None:
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_len = max_len

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict:
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            ).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        label_id = int(np.argmax(probs))
        return {
            "label": LABELS[label_id],
            "label_id": label_id,
            "confidence": float(probs[label_id]),
            "scores": {"negative": float(probs[0]), "positive": float(probs[1])},
        }

    def predict_batch(
        self, texts: List[str], batch_size: int = 64, show_progress: bool = True
    ) -> List[Dict]:
        results: List[Dict] = []
        batches = range(0, len(texts), batch_size)
        if show_progress:
            batches = tqdm(batches, desc="Inference", unit="batch")

        for start in batches:
            batch_texts = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch_texts,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                ).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for p in probs:
                label_id = int(np.argmax(p))
                results.append({
                    "label": LABELS[label_id],
                    "label_id": label_id,
                    "confidence": float(p[label_id]),
                    "scores": {"negative": float(p[0]), "positive": float(p[1])},
                })
        return results
