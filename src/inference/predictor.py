import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Optional

LABELS = {0: "negative", 1: "positive"}


class SentimentPredictor:
    def __init__(
        self,
        model_dir: str = "models/roberta-fitness-binary",
        device: Optional[str] = None,
        max_len: int = 128,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict:
        enc = self.tokenizer(
            text.lower(),
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
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> List[Dict]:
        results: List[Dict] = []
        indices = range(0, len(texts), batch_size)
        if show_progress:
            indices = tqdm(indices, desc="Inference", unit="batch")
        for start in indices:
            enc = self.tokenizer(
                [t.lower() for t in texts[start : start + batch_size]],
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
