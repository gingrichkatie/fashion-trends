from __future__ import annotations

import numpy as np
import pandas as pd


def generate_demo_panel(seed: int = 42):
    np.random.seed(seed)
    # Use actual items from training data for more realistic predictions
    items = [
        "Backpack",
        "Belt",
        "Blazer",
        "Blouse",
        "Boots",
        "Dress",
        "Jacket",
        "Jeans",
        "Sneakers",
        "T-shirt",
    ]
    start = pd.Timestamp("2023-01-01").to_period("M").to_timestamp()
    months = pd.date_range(start, periods=14, freq="MS")

    rows = []
    for item in items:
        # More realistic base values matching training data distribution (mean ~715, wide range)
        base = np.random.uniform(200, 1200)  # Wider range like training data
        noise = np.random.normal(0, 100, size=len(months))  # More realistic variance
        seasonal = 150 * np.sin(2 * np.pi * (np.arange(len(months)) / 12.0))
        ts = np.clip(
            base + seasonal + noise, 50, 3000
        )  # Allow wider range like training
        orders = np.clip(
            (ts / 45 + np.random.normal(0, 3, len(ts))).round(), 1, 25
        )  # More realistic order counts
        rating = np.clip(np.random.normal(4.2, 0.3, len(ts)), 3.0, 5.0)
        payshare = np.clip(np.random.beta(5, 2, size=len(ts)), 0, 1)
        for m, sales, n, r, ps in zip(months, ts, orders, rating, payshare):
            rows.append([item, m, float(sales), int(n), float(r), float(ps)])

    df = pd.DataFrame(
        rows,
        columns=[
            "Item Purchased",
            "month_start",
            "total_sales",
            "orders",
            "avg_rating",
            "pay_ref_share",
        ],
    )
    df["year"] = df["month_start"].dt.year
    df["month"] = df["month_start"].dt.month
    anchor = df["month_start"].min()
    df["ym_idx"] = (df["month_start"].dt.to_period("M") - anchor.to_period("M")).apply(
        lambda p: p.n
    )

    df = df.sort_values(["Item Purchased", "month_start"]).reset_index(drop=True)
    g = df.groupby("Item Purchased", sort=False)
    df["total_sales_lag1"] = g["total_sales"].shift(1)
    df["orders_lag1"] = g["orders"].shift(1)
    df["total_sales_roll3"] = g["total_sales"].shift(1).rolling(3).mean()
    df["orders_roll3"] = g["orders"].shift(1).rolling(3).mean()
    df = df.dropna(
        subset=["total_sales_lag1", "orders_lag1", "total_sales_roll3", "orders_roll3"]
    ).reset_index(drop=True)
    return df, anchor
