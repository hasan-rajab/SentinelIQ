"""
SentinelIQ — SHAP Explainer
Generates SHAP-based feature attributions for anomalies detected
by IsolationForest and Autoencoder models.
"""

import numpy as np
import pandas as pd
import json
from typing import Optional


class SentinelShapExplainer:
    """
    Wraps SHAP explainers for IsolationForest and Autoencoder.
    Produces per-sample feature attributions for anomalous records.
    """

    def __init__(self, model_type: str, feature_cols: list):
        """
        model_type: 'isolation_forest' or 'autoencoder'
        """
        self.model_type = model_type
        self.feature_cols = feature_cols
        self.explainer = None

    def fit(self, model, X_background: np.ndarray):
        """
        Fit the SHAP explainer on background data.
        X_background: numpy array of normal samples (scaled).
        """
        import shap

        if self.model_type == "isolation_forest":
            # Use TreeExplainer for sklearn IF
            self.explainer = shap.TreeExplainer(model)

        elif self.model_type == "autoencoder":
            # Use DeepExplainer for PyTorch autoencoder
            import torch

            background = torch.FloatTensor(X_background[:100])  # limit background size

            def model_fn(x):
                import torch
                model.eval()
                with torch.no_grad():
                    recon = model(torch.FloatTensor(x))
                    # Return reconstruction error per sample
                    return torch.mean((recon - torch.FloatTensor(x)) ** 2, dim=1).numpy()

            self.explainer = shap.KernelExplainer(model_fn, X_background[:50])

        print(f"[ShapExplainer] Fitted for {self.model_type}")
        return self

    def explain(self, X: np.ndarray, max_samples: int = 100) -> np.ndarray:
        """
        Returns SHAP values array of shape (n_samples, n_features).
        """
        import shap

        X_sample = X[:max_samples]

        if self.model_type == "isolation_forest":
            shap_values = self.explainer.shap_values(X_sample)
            # IF returns list for multi-output — take anomaly class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        elif self.model_type == "autoencoder":
            shap_values = self.explainer.shap_values(X_sample)

        return shap_values

    def explain_single(self, x: np.ndarray) -> dict:
        """
        Explain a single sample. Returns feature → shap_value dict,
        sorted by absolute importance.
        """
        shap_vals = self.explain(x.reshape(1, -1))[0]
        attribution = {
            col: round(float(val), 6)
            for col, val in zip(self.feature_cols, shap_vals)
        }
        return dict(sorted(attribution.items(), key=lambda kv: abs(kv[1]), reverse=True))

    def top_features(self, x: np.ndarray, n: int = 3) -> list:
        """Returns top-n most important features for a single sample."""
        attribution = self.explain_single(x)
        return list(attribution.keys())[:n]

    def build_explanation(
        self,
        record: dict,
        shap_attribution: dict,
        mitre_mapping: dict,
        fused_score: float,
    ) -> dict:
        """
        Builds a complete human-readable explanation for an anomaly.
        """
        top_features = list(shap_attribution.keys())[:3]
        top_values = {k: record.get(k, "N/A") for k in top_features}

        explanation = {
            "fused_score": round(fused_score, 4),
            "severity": mitre_mapping.get("severity", "unknown"),
            "mitre_tactic": mitre_mapping.get("tactic", "Unknown"),
            "mitre_tactic_id": mitre_mapping.get("tactic_id", "N/A"),
            "mitre_technique": mitre_mapping.get("technique", "Unknown"),
            "mitre_technique_id": mitre_mapping.get("technique_id", "N/A"),
            "description": mitre_mapping.get("description", ""),
            "recommended_action": mitre_mapping.get("recommended_action", ""),
            "top_features": top_features,
            "feature_values": top_values,
            "shap_attribution": shap_attribution,
            "anomaly_type": record.get("anomaly_type", "unknown"),
            "source": record.get("source", record.get("host", record.get("src_ip", "unknown"))),
            "timestamp": record.get("timestamp", ""),
        }

        return explanation

    def narrative(self, explanation: dict) -> str:
        """
        Generates a plain-English narrative for an anomaly explanation.
        Used as input to the Groq LLM narration layer in Phase 3.
        """
        top = explanation["top_features"]
        vals = explanation["feature_values"]
        feat_str = ", ".join([f"{f}={vals[f]}" for f in top])

        return (
            f"A {explanation['severity'].upper()} severity anomaly was detected on "
            f"{explanation['source']} at {explanation['timestamp']}. "
            f"The anomaly score is {explanation['fused_score']:.3f}. "
            f"This matches the MITRE ATT&CK technique {explanation['mitre_technique']} "
            f"({explanation['mitre_technique_id']}) under tactic {explanation['mitre_tactic']} "
            f"({explanation['mitre_tactic_id']}). "
            f"The top contributing features are: {feat_str}. "
            f"Description: {explanation['description']} "
            f"Recommended action: {explanation['recommended_action']}"
        )