from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib

# sklearn compat shim for legacy pickles
try:
    import sklearn.compose._column_transformer as _ct  # type: ignore[attr-defined]

    if not hasattr(_ct, "_RemainderColsList"):

        class _RemainderColsList(list): ...

        _ct._RemainderColsList = _RemainderColsList  # type: ignore[attr-defined]
except Exception:
    pass

ARTIFACTS_DIRNAME = "artifacts"
FN_PIPELINE = "fashion_sales_monthly_pipeline.pkl"


class SklearnModelRepository:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)

    def load(self) -> Tuple[Any, None, Dict]:
        """Load the pipeline artifact."""
        artifacts_dir = self.base_dir / ARTIFACTS_DIRNAME
        pipeline_path = artifacts_dir / FN_PIPELINE

        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"Pipeline not found: {pipeline_path}\n"
                f"Expected: {ARTIFACTS_DIRNAME}/{FN_PIPELINE}"
            )

        if pipeline_path.stat().st_size < 500:
            raise ValueError(
                f"Pipeline file {FN_PIPELINE} appears corrupted (too small)"
            )

        pipeline = joblib.load(pipeline_path)
        config = {
            "ARTIFACT_TYPE": "pipeline",
            "PIPELINE_FILE": str(pipeline_path),
            "target_column": "total_sales",
        }

        return pipeline, None, config
