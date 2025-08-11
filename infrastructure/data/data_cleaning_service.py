"""
Data cleaning service for preparing data for visualization.

This service encapsulates all data cleaning logic, ensuring that data
passed to the presentation layer is clean and safe for visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


class DataCleaningService:
    """Service responsible for cleaning and preparing data for visualization."""

    def __init__(self):
        self.cleaning_stats = {}

    def clean_for_visualization(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean DataFrame for safe visualization in charts.

        Args:
            data: Raw DataFrame that may contain infinite or NaN values

        Returns:
            Cleaned DataFrame safe for visualization
        """
        if data.empty:
            return data

        cleaned_data = data.copy()

        # Track cleaning operations for debugging
        self.cleaning_stats = {
            "original_rows": len(data),
            "numeric_columns": [],
            "infinite_values_found": 0,
            "nan_values_found": 0,
            "rows_dropped": 0,
            "cleaning_operations": [],
        }

        # Identify numeric columns
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        self.cleaning_stats["numeric_columns"] = list(numeric_cols)

        # Clean numeric columns
        for col in numeric_cols:
            cleaned_data[col] = self._clean_numeric_column(cleaned_data[col], col)

        # Drop rows with NaN in critical columns
        critical_cols = self._identify_critical_columns(cleaned_data)
        if critical_cols:
            initial_rows = len(cleaned_data)
            cleaned_data = cleaned_data.dropna(subset=critical_cols)
            rows_dropped = initial_rows - len(cleaned_data)
            self.cleaning_stats["rows_dropped"] = rows_dropped

            if rows_dropped > 0:
                self.cleaning_stats["cleaning_operations"].append(
                    f"Dropped {rows_dropped} rows with NaN in critical columns: {critical_cols}"
                )

        self.cleaning_stats["final_rows"] = len(cleaned_data)

        return cleaned_data

    def _clean_numeric_column(self, series: pd.Series, col_name: str) -> pd.Series:
        """Clean a single numeric column."""
        # Convert to numeric, coercing errors to NaN
        cleaned_series = pd.to_numeric(series, errors="coerce")

        # Count issues before cleaning
        inf_count = np.isinf(cleaned_series).sum()
        nan_count = np.isnan(cleaned_series).sum()

        self.cleaning_stats["infinite_values_found"] += inf_count
        self.cleaning_stats["nan_values_found"] += nan_count

        if inf_count > 0:
            self.cleaning_stats["cleaning_operations"].append(
                f"Replaced {inf_count} infinite values in '{col_name}' with 0"
            )

        # Replace infinite values with 0
        cleaned_series = cleaned_series.replace([np.inf, -np.inf], 0)

        # Fill remaining NaN values with 0 for visualization safety
        cleaned_series = cleaned_series.fillna(0)

        return cleaned_series

    def _identify_critical_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify columns that are critical for visualization."""
        critical_patterns = [
            "sales",
            "actual",
            "predicted",
            "month",
            "date",
            "item",
            "style",
        ]

        critical_cols = []
        for col in data.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in critical_patterns):
                critical_cols.append(col)

        return critical_cols

    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Get summary of last cleaning operation."""
        return self.cleaning_stats.copy()

    def has_data_quality_issues(self) -> bool:
        """Check if the last cleaning operation found data quality issues."""
        return (
            self.cleaning_stats.get("infinite_values_found", 0) > 0
            or self.cleaning_stats.get("nan_values_found", 0) > 0
            or self.cleaning_stats.get("rows_dropped", 0) > 0
        )

    def format_cleaning_report(self) -> str:
        """Format a human-readable cleaning report."""
        stats = self.cleaning_stats

        if not self.has_data_quality_issues():
            return f"✅ Data is clean: {stats.get('final_rows', 0)} rows ready for visualization"

        issues = []
        if stats.get("infinite_values_found", 0) > 0:
            issues.append(f"{stats['infinite_values_found']} infinite values")
        if stats.get("nan_values_found", 0) > 0:
            issues.append(f"{stats['nan_values_found']} NaN values")
        if stats.get("rows_dropped", 0) > 0:
            issues.append(f"{stats['rows_dropped']} problematic rows")

        issue_text = ", ".join(issues)

        return (
            f"⚠️ Data cleaned: Found and fixed {issue_text}. "
            f"{stats.get('final_rows', 0)} rows ready for visualization."
        )
