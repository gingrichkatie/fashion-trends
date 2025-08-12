from __future__ import annotations

from typing import Tuple
import numpy as np


def compute_pi_q(y_true, y_pred, target_coverage: float = 0.80) -> float:
    resid = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    return float(np.quantile(resid, target_coverage))


def interval_from_q(yhat, q: float) -> Tuple[np.ndarray, np.ndarray]:
    yhat = np.asarray(yhat, dtype=float)
    low = np.maximum(0, yhat - q)
    high = yhat + q
    return low, high
