from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def _fit_linear(y_true: np.ndarray, y_hat: np.ndarray) -> Tuple[float, float, float]:
    """Fit y_true ≈ a*y_hat + b using least squares; return (a, b, rmse)."""
    X = np.column_stack([y_hat, np.ones_like(y_hat)])
    try:
        coeffs, *_ = np.linalg.lstsq(X, y_true, rcond=None)
        a, b = float(coeffs[0]), float(coeffs[1])
    except Exception:
        a, b = 1.0, 0.0
    y_cal = a * y_hat + b
    rmse = float(np.sqrt(np.mean((y_true - y_cal) ** 2)))
    return a, b, rmse


def detect_output_scale(
    panel: pd.DataFrame, svc
) -> Tuple[str, Dict[str, float], Tuple[float, float]]:
    """Choose scale among {dollars, log, log1p} using linear-calibrated RMSE.

    Returns (mode, raw_rmses, calib_ab)
      - mode: chosen scale name
      - raw_rmses: RMSE without linear calibration for each candidate
      - calib_ab: (a, b) linear calibration for the chosen mode
    """
    y_true = panel[svc.target_column_name].astype(float).values
    raw = svc.predict_raw(panel).astype(float)
    candidates = {
        "dollars": raw,
        "log": np.exp(raw),
        "log1p": np.expm1(raw),
    }
    raw_rmses: Dict[str, float] = {
        k: float(np.sqrt(np.mean((y_true - v) ** 2))) for k, v in candidates.items()
    }
    calib_scores: Dict[str, Tuple[float, float, float]] = {}
    for k, v in candidates.items():
        calib_scores[k] = _fit_linear(y_true, v)
    # choose by calibrated rmse
    mode = min(calib_scores.keys(), key=lambda k: calib_scores[k][2])
    a, b, _ = calib_scores[mode]
    return mode, raw_rmses, (a, b)
