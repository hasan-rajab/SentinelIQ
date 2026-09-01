"""Decision-model-aligned feature attribution helpers.

The goal is not to label every explanation as SHAP. Each helper explains the
actual model signal used by live inference so the UI never attributes a
decision to a different model than the one that produced it.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import torch


def autoencoder_reconstruction_attribution(model, record: dict) -> Dict[str, float]:
    """Return per-feature squared reconstruction error for one record."""
    df = pd.DataFrame([record])
    x = model.scaler.transform(df[model.feature_cols].fillna(0).values)
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=model.device)

    model.net.eval()
    with torch.no_grad():
        reconstruction = model.net(x_tensor)
        errors = ((reconstruction - x_tensor) ** 2).detach().cpu().numpy()[0]

    return {
        feature: float(error)
        for feature, error in sorted(
            zip(model.feature_cols, errors),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    }


def xgboost_contribution_attribution(model, df: pd.DataFrame) -> Dict[str, float]:
    """Return XGBoost per-feature margin contributions for one record.

    Uses the booster-native ``pred_contribs`` path. The final bias term is
    intentionally omitted; only feature contributions are returned. The model
    was trained from NumPy arrays, so the DMatrix intentionally has no feature
    names; contributions are mapped back by their stable training-column order.
    """
    from xgboost import DMatrix

    x = model.scaler.transform(model._extract(df))
    contributions = model.model.get_booster().predict(
        DMatrix(x),
        pred_contribs=True,
    )[0][:-1]

    return {
        feature: float(value)
        for feature, value in sorted(
            zip(model.feature_cols, contributions),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    }


def top_features(attribution: Dict[str, float], limit: int = 3) -> list[str]:
    return list(attribution.keys())[:limit]
