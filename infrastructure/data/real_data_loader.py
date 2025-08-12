"""
Real data loader that replicates the exact aggregation process from the Jupyter notebook.
This ensures we use the same data the model was trained on.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np


class RealDataLoader:
    """Loads and processes real Fashion Retail Sales data using the exact notebook process."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = Path(csv_path)

    def load_and_process(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load raw data and process it exactly like the notebook.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, full_panel)
        """
        # Load raw data
        raw = pd.read_csv(self.csv_path)

        # Parse dates (day-first like 23-03-2023)
        raw["Date Purchase"] = pd.to_datetime(
            raw["Date Purchase"], errors="coerce", dayfirst=True
        )
        raw = raw.dropna(subset=["Date Purchase"]).copy()

        # Month start for grouping
        raw["month_start"] = raw["Date Purchase"].dt.to_period("M").dt.to_timestamp()

        # Choose reference payment method
        top_method = (
            raw["Payment Method"].mode().iat[0]
            if not raw["Payment Method"].mode().empty
            else None
        )
        ref_method = (
            "Credit Card"
            if "Credit Card" in raw["Payment Method"].unique()
            else top_method
        )

        # Aggregate to item × month
        panel = raw.groupby(["Item Purchased", "month_start"], as_index=False).agg(
            total_sales=("Purchase Amount (USD)", "sum"),
            orders=("Purchase Amount (USD)", "size"),
            avg_rating=("Review Rating", "mean"),
            pay_ref_share=(
                "Payment Method",
                lambda s: (s == ref_method).mean() if ref_method else 0.0,
            ),
        )

        # Calendar features
        panel["year"] = panel["month_start"].dt.year
        panel["month"] = panel["month_start"].dt.month
        panel["ym_idx"] = (panel["year"] - panel["year"].min()) * 12 + panel["month"]

        panel.sort_values(["Item Purchased", "month_start"], inplace=True)

        # Enhanced Feature Engineering for High Performance
        g = panel.groupby("Item Purchased", sort=False)

        # === BASIC LAGS & ROLLING FEATURES ===
        # 1-step lag of target and orders
        panel["total_sales_lag1"] = g["total_sales"].shift(1)
        panel["orders_lag1"] = g["orders"].shift(1)

        # Multiple lag periods for trend analysis
        panel["total_sales_lag2"] = g["total_sales"].shift(2)
        panel["total_sales_lag3"] = g["total_sales"].shift(3)
        panel["orders_lag2"] = g["orders"].shift(2)
        panel["orders_lag3"] = g["orders"].shift(3)

        # Rolling statistics (use only past info: shift before rolling)
        panel["total_sales_roll3"] = g["total_sales"].shift(1).rolling(3).mean()
        panel["orders_roll3"] = g["orders"].shift(1).rolling(3).mean()
        panel["total_sales_roll6"] = g["total_sales"].shift(1).rolling(6).mean()
        panel["orders_roll6"] = g["orders"].shift(1).rolling(6).mean()

        # Rolling std for volatility features
        panel["total_sales_std3"] = g["total_sales"].shift(1).rolling(3).std()
        panel["orders_std3"] = g["orders"].shift(1).rolling(3).std()

        # === TREND ANALYSIS ===
        # Growth rates (month-over-month)
        panel["sales_growth_1m"] = (
            panel["total_sales_lag1"] - panel["total_sales_lag2"]
        ) / (panel["total_sales_lag2"] + 1)
        panel["orders_growth_1m"] = (panel["orders_lag1"] - panel["orders_lag2"]) / (
            panel["orders_lag2"] + 1
        )

        # 3-month trend (comparing recent vs older performance)
        panel["sales_trend_3m"] = (
            panel["total_sales_roll3"]
            - panel.groupby("Item Purchased")["total_sales"].transform(
                lambda x: x.shift(4).rolling(3).mean()
            )
        ) / (
            panel.groupby("Item Purchased")["total_sales"].transform(
                lambda x: x.shift(4).rolling(3).mean()
            )
            + 1
        )

        # === SEASONALITY FEATURES ===
        # Quarter and season
        panel["quarter"] = ((panel["month"] - 1) // 3) + 1
        panel["season"] = panel["month"].map(
            {
                12: "Winter",
                1: "Winter",
                2: "Winter",
                3: "Spring",
                4: "Spring",
                5: "Spring",
                6: "Summer",
                7: "Summer",
                8: "Summer",
                9: "Fall",
                10: "Fall",
                11: "Fall",
            }
        )

        # Cyclical features for seasonality
        panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12)
        panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12)
        panel["quarter_sin"] = np.sin(2 * np.pi * panel["quarter"] / 4)
        panel["quarter_cos"] = np.cos(2 * np.pi * panel["quarter"] / 4)

        # === INTERACTION FEATURES ===
        # Price per order (revenue efficiency)
        panel["price_per_order"] = panel["total_sales"] / (panel["orders"] + 1)
        panel["price_per_order_lag1"] = g["price_per_order"].shift(1)

        # Rating interaction with sales/orders
        panel["rating_sales_interaction"] = (
            panel["avg_rating"] * panel["total_sales_lag1"]
        )
        panel["rating_orders_interaction"] = panel["avg_rating"] * panel["orders_lag1"]

        # Payment preference interaction
        panel["pay_sales_interaction"] = (
            panel["pay_ref_share"] * panel["total_sales_lag1"]
        )

        # === ITEM-LEVEL FEATURES ===
        # Item performance relative to its own history (use transform to ensure index alignment)
        panel["sales_vs_item_avg"] = panel["total_sales_lag1"] / (
            panel.groupby("Item Purchased")["total_sales"].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            + 1
        )
        panel["orders_vs_item_avg"] = panel["orders_lag1"] / (
            panel.groupby("Item Purchased")["orders"].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            + 1
        )

        # Item ranking features (performance percentile within month)
        monthly_groups = panel.groupby(["year", "month"])
        panel["sales_rank_in_month"] = monthly_groups["total_sales_lag1"].rank(pct=True)
        panel["orders_rank_in_month"] = monthly_groups["orders_lag1"].rank(pct=True)

        # === TIME-BASED FEATURES ===
        # Time since first sale for this item
        panel["months_since_first_sale"] = g.cumcount()

        # Boolean indicators
        panel["is_first_3_months"] = (panel["months_since_first_sale"] < 3).astype(int)
        panel["is_high_rating"] = (panel["avg_rating"] > 4.0).astype(int)
        panel["is_premium_item"] = (
            panel["price_per_order_lag1"] > panel["price_per_order_lag1"].quantile(0.75)
        ).astype(int)

        # Drop first rows with missing lags/rolls (now need more due to additional lags)
        panel = panel.dropna(
            subset=[
                "total_sales_lag1",
                "orders_lag1",
                "total_sales_roll3",
                "orders_roll3",
                "total_sales_lag3",
                "orders_lag3",  # Ensure we have at least 3 lags
            ]
        ).reset_index(drop=True)

        # Define enhanced feature set
        TARGET = "total_sales"
        num_cols = [
            # Original features
            "orders",
            "avg_rating",
            "pay_ref_share",
            "ym_idx",
            # Basic lags and rolling
            "total_sales_lag1",
            "total_sales_lag2",
            "total_sales_lag3",
            "orders_lag1",
            "orders_lag2",
            "orders_lag3",
            "total_sales_roll3",
            "total_sales_roll6",
            "orders_roll3",
            "orders_roll6",
            "total_sales_std3",
            "orders_std3",
            # Trend and growth
            "sales_growth_1m",
            "orders_growth_1m",
            "sales_trend_3m",
            # Seasonality (cyclical)
            "month_sin",
            "month_cos",
            "quarter_sin",
            "quarter_cos",
            # Interactions and ratios
            "price_per_order",
            "price_per_order_lag1",
            "rating_sales_interaction",
            "rating_orders_interaction",
            "pay_sales_interaction",
            # Item-level performance
            "sales_vs_item_avg",
            "orders_vs_item_avg",
            "sales_rank_in_month",
            "orders_rank_in_month",
            # Time features
            "months_since_first_sale",
            "is_first_3_months",
            "is_high_rating",
            "is_premium_item",
        ]

        cat_cols = ["Item Purchased", "year", "month", "quarter", "season"]

        X = panel[num_cols + cat_cols].copy()
        y = panel[TARGET].astype(float)

        # Split by month (80/20 split like in notebook)
        unique_months = panel["month_start"].sort_values().unique()
        cut = int(0.8 * len(unique_months))
        cutoff_month = unique_months[cut]

        mask = panel["month_start"] < cutoff_month
        X_train, X_test = X[mask], X[~mask]
        y_train, y_test = y[mask], y[~mask]

        return X_train, X_test, y_train, y_test, panel

    def get_validation_panel(self) -> pd.DataFrame:
        """Get the test/validation panel for model evaluation."""
        _, X_test, _, y_test, full_panel = self.load_and_process()

        # Reconstruct the test panel with all necessary columns for visualization
        test_indices = X_test.index
        validation_panel = full_panel.loc[test_indices].copy()

        return validation_panel

    def get_full_panel(self) -> pd.DataFrame:
        """Get the complete processed panel for comprehensive evaluation."""
        _, _, _, _, full_panel = self.load_and_process()
        return full_panel

    def get_data_summary(self) -> dict:
        """Get summary statistics about the real data."""
        _, _, _, _, panel = self.load_and_process()

        return {
            "total_rows": len(panel),
            "unique_items": panel["Item Purchased"].nunique(),
            "date_range": f"{panel['month_start'].min()} to {panel['month_start'].max()}",
            "total_sales_stats": {
                "mean": float(panel["total_sales"].mean()),
                "std": float(panel["total_sales"].std()),
                "min": float(panel["total_sales"].min()),
                "max": float(panel["total_sales"].max()),
                "median": float(panel["total_sales"].median()),
            },
            "items": sorted(panel["Item Purchased"].unique().tolist()),
        }
