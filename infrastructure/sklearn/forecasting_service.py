from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pandas as pd


class SklearnForecastingService:
    def __init__(self, model: Any, preprocess: Any, config: Dict) -> None:
        self._pipeline = model  # This is actually the pipeline now
        self._config = config

    @property
    def target_column_name(self) -> str:
        return str(self._config.get("target_column", "total_sales"))

    def predict_in_dollars(self, df: pd.DataFrame) -> np.ndarray:
        """Predict directly in dollars using the pipeline."""
        return np.asarray(self._pipeline.predict(df)).astype(float)
