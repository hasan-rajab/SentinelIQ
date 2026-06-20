"""
SentinelIQ — BERT Log Anomaly Detector
Fine-tunes BERT for binary classification of log lines:
  0 = normal, 1 = anomaly
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix,
)
from torch.optim import AdamW


# ── Dataset ───────────────────────────────────────────────────────────────────
class LogDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


# ── Model wrapper ─────────────────────────────────────────────────────────────
class SentinelBertLog:
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("model_name", "bert-base-uncased")
        self.max_length = config.get("max_length", 128)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = config.get("threshold", 0.5)
        self.is_fitted = False

        print(f"[BertLog] Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=config.get("num_labels", 2),
        ).to(self.device)

    def _build_loader(self, texts: list, labels: list, shuffle: bool) -> DataLoader:
        dataset = LogDataset(texts, labels, self.tokenizer, self.max_length)
        return DataLoader(dataset, batch_size=self.config.get("batch_size", 32), shuffle=shuffle)

    def fit(self, df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        texts = df["message"].tolist()
        labels = df["is_anomaly"].astype(int).tolist()
        loader = self._build_loader(texts, labels, shuffle=True)

        epochs = self.config.get("epochs", 5)
        total_steps = len(loader) * epochs
        warmup_steps = int(total_steps * self.config.get("warmup_ratio", 0.1))

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 2e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        # Class weighting for imbalanced data
        n_normal = (df["is_anomaly"] == False).sum()
        n_anomaly = (df["is_anomaly"] == True).sum()
        weight = torch.FloatTensor([1.0, n_normal / max(n_anomaly, 1)]).to(self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)

        print(f"[BertLog] Training | samples={len(df)} | device={self.device} | epochs={epochs}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0
            correct = 0

            for batch in loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                correct += (outputs.logits.argmax(dim=1) == batch_labels).sum().item()

            acc = correct / len(df)
            print(f"  Epoch {epoch}/{epochs} | loss={total_loss/len(loader):.4f} | acc={acc:.4f}")

            if val_df is not None:
                val_metrics = self.evaluate(val_df, val_df["is_anomaly"].astype(int).values)
                print(f"  Val F1={val_metrics['f1']} | ROC-AUC={val_metrics['roc_auc']}")

        self.is_fitted = True
        return self

    def predict_proba(self, texts: list) -> np.ndarray:
        self.model.eval()
        all_probs = []
        batch_size = self.config.get("batch_size", 32)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

        return np.array(all_probs)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(df["message"].tolist())

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.score(df) >= self.threshold).astype(int)

    def evaluate(self, df: pd.DataFrame, y_true: np.ndarray) -> dict:
        scores = self.score(df)
        y_pred = (scores >= self.threshold).astype(int)

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        try:
            roc_auc = roc_auc_score(y_true, scores)
            avg_precision = average_precision_score(y_true, scores)
        except ValueError:
            roc_auc = avg_precision = 0.0

        metrics = {
            "roc_auc": round(roc_auc, 4),
            "avg_precision": round(avg_precision, 4),
            "precision": round(report.get("1", {}).get("precision", 0), 4),
            "recall": round(report.get("1", {}).get("recall", 0), 4),
            "f1": round(report.get("1", {}).get("f1-score", 0), 4),
            "accuracy": round(report.get("accuracy", 0), 4),
            "confusion_matrix": cm.tolist(),
            "threshold": self.threshold,
            "n_samples": len(df),
            "n_anomalies_true": int(y_true.sum()),
            "n_anomalies_pred": int(y_pred.sum()),
        }

        print(f"\n[BertLog] Evaluation Results:")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"  {k:20}: {v}")

        return metrics

    def save(self, save_dir: str, name: str = "bert_log"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(f"{save_dir}/{name}")
        self.tokenizer.save_pretrained(f"{save_dir}/{name}")
        meta = {"threshold": self.threshold, "config": self.config}
        with open(f"{save_dir}/{name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[BertLog] Saved to {save_dir}/{name}/")

    @classmethod
    def load(cls, save_dir: str, name: str = "bert_log") -> "SentinelBertLog":
        with open(f"{save_dir}/{name}_meta.json") as f:
            meta = json.load(f)
        obj = cls(config=meta["config"])
        obj.model = BertForSequenceClassification.from_pretrained(f"{save_dir}/{name}").to(obj.device)
        obj.tokenizer = AutoTokenizer.from_pretrained(f"{save_dir}/{name}")
        obj.threshold = meta["threshold"]
        obj.is_fitted = True
        print(f"[BertLog] Loaded from {save_dir}/{name}/")
        return obj