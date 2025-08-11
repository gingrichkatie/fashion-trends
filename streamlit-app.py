from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Ensure project root is importable BEFORE application/infrastructure imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Dependency injection imports
from infrastructure.sklearn.model_repository import SklearnModelRepository
from infrastructure.sklearn.forecasting_service import SklearnForecastingService
from infrastructure.data.real_data_loader import RealDataLoader
from infrastructure.data.data_cleaning_service import DataCleaningService
from core.use_cases.intervals import compute_pi_q, interval_from_q


st.set_page_config(
    page_title="Fashion Analytics Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_services(base_dir: Path) -> Tuple[SklearnForecastingService, Dict]:
    repo = SklearnModelRepository(base_dir)
    model, preprocess, cfg = repo.load()
    svc = SklearnForecastingService(model=model, preprocess=preprocess, config=cfg)
    return svc, cfg


@st.cache_data(show_spinner=False)
def cached_real_data(csv_path: str):
    """Load and cache real data processing."""
    loader = RealDataLoader(Path(csv_path))
    full_panel = loader.get_full_panel()  # Use full dataset for visualization
    validation_panel = (
        loader.get_validation_panel()
    )  # Keep validation for model testing
    data_summary = loader.get_data_summary()
    return full_panel, validation_panel, data_summary


def rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "Item Purchased": "Item / Style",
        "year": "Year",
        "month": "Month",
        "ym_idx": "Month index",
        "orders": "Orders",
        "avg_rating": "Average rating (1–5)",
        "pay_ref_share": "Preferred pay share (0–1)",
        "total_sales_lag1": "Last month sales ($)",
        "total_sales_roll3": "3-month avg sales ($)",
        "orders_lag1": "Last month orders",
        "orders_roll3": "3-month avg orders",
        "total_sales": "Actual sales ($)",
        "pred_total_sales": "Predicted sales ($)",
        "pi_low": "PI low ($)",
        "pi_high": "PI high ($)",
        "month_start": "Month start",
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def main():
    base = Path(__file__).resolve().parent

    # Custom CSS for pink theme and improved styling
    st.markdown(
        """
    <style>
        .main-header {
            background: linear-gradient(90deg, #ff6b9d, #ffc1cc);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .main-header h1 {
            color: white;
            margin: 0;
            font-size: 2.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .main-header p {
            color: white;
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        .metric-compact {
            background: linear-gradient(135deg, #fff0f5, #ffe4e6);
            padding: 0.8rem;
            border-radius: 8px;
            border-left: 4px solid #ff6b9d;
            margin: 0.2rem 0;
        }
        .status-good { border-left-color: #10b981; }
        .status-warning { border-left-color: #f59e0b; }
        .status-error { border-left-color: #ef4444; }
        .insight-box {
            background: linear-gradient(135deg, #fef7ff, #fce7f3);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #f9a8d4;
            margin: 1rem 0;
        }
        .section-divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #ff6b9d, transparent);
            margin: 2rem 0;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Compact header
    st.markdown(
        """
    <div class="main-header">
        <h1>🌸 Fashion Analytics Dashboard</h1>
        <p>AI-powered sales insights and forecasting</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load artifacts and services
    try:
        svc, cfg = load_services(base)
        artifacts_ok = True
    except Exception as e:
        svc = None
        cfg = {}
        artifacts_ok = False
        st.warning(f"Artifacts failed to load: {e}")

    # Real data panel
    csv_path = base / "data" / "raw" / "Fashion_Retail_Sales.csv"
    try:
        full_panel_df, validation_panel_df, data_summary = cached_real_data(
            str(csv_path)
        )
        data_ok = True
        # Use full panel for visualization, validation panel for model testing
        panel_df = full_panel_df
    except Exception as e:
        panel_df = pd.DataFrame()
        validation_panel_df = pd.DataFrame()
        data_summary = {}
        data_ok = False
        st.warning(f"Real data failed to load: {e}")

    # Score panel & intervals
    rmse_panel = None
    q = None
    y_pred = None
    baseline_rmse_naive1 = None
    baseline_rmse_roll3 = None

    if artifacts_ok and data_ok and svc is not None:
        # Use validation panel for model performance metrics (clean first)
        validation_metrics_cleaning = DataCleaningService()
        validation_cleaned_metrics = (
            validation_metrics_cleaning.clean_for_visualization(validation_panel_df)
        )
        y_true = validation_cleaned_metrics[svc.target_column_name].astype(float).values
        y_pred = svc.predict_in_dollars(validation_cleaned_metrics)
        rmse_panel = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        q = compute_pi_q(y_true, y_pred, target_coverage=0.80)

        # Baselines for context
        def rmse(a, b):
            m = (~np.isnan(a)) & (~np.isnan(b))
            if m.sum() == 0:
                return None
            return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

        naive1 = validation_panel_df["total_sales_lag1"].astype(float).values
        roll3 = validation_panel_df["total_sales_roll3"].astype(float).values
        baseline_rmse_naive1 = rmse(y_true, naive1)
        baseline_rmse_roll3 = rmse(y_true, roll3)

    if not data_ok:
        st.error("Real data not available - cannot display charts")
        # Still show basic status
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        with col1:
            st.markdown(
                """
            <div class="metric-compact status-error">
                <strong>💰 Total Sales</strong><br/>
                Data unavailable
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
            <div class="metric-compact status-error">
                <strong>📦 Coverage</strong><br/>
                No data loaded
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                """
            <div class="metric-compact status-error">
                <strong>System Status</strong><br/>
                🔴 Data Missing
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                """
            <div class="metric-compact status-error">
                <strong>⚠️ Model Performance</strong><br/>
                Cannot evaluate
            </div>
            """,
                unsafe_allow_html=True,
            )
        return

    disp = rename_for_display(panel_df)

    # Use data cleaning service to prepare data for visualization
    cleaning_service = DataCleaningService()

    disp_clean = cleaning_service.clean_for_visualization(disp)

    # Show cleaning report
    if cleaning_service.has_data_quality_issues():
        st.warning(cleaning_service.format_cleaning_report())
    else:
        st.success(cleaning_service.format_cleaning_report())

    # Validate that we have clean data
    if len(disp_clean) == 0:
        st.error("❌ No valid data remaining after cleaning. Cannot display charts.")
        return

    # Compact status row (after disp is defined)
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        total_sales = disp_clean["Actual sales ($)"].sum()
        st.markdown(
            f"""
        <div class="metric-compact">
            <strong>💰 Total Sales</strong><br/>
            ${total_sales:,.0f}
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        items_count = (
            data_summary.get("unique_items", "—")
            if data_summary
            else disp_clean["Item / Style"].nunique()
        )
        months_count = len(disp_clean["Month start"].unique())
        st.markdown(
            f"""
        <div class="metric-compact">
            <strong>📦 Coverage</strong><br/>
            {items_count} items, {months_count} months
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        status_class = "status-good" if (artifacts_ok and data_ok) else "status-error"
        status_text = (
            "🟢 All Systems OK" if (artifacts_ok and data_ok) else "🔴 Issues Detected"
        )
        st.markdown(
            f"""
        <div class="metric-compact {status_class}">
            <strong>System Status</strong><br/>
            {status_text}
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        if rmse_panel and artifacts_ok and data_ok and svc:
            # Calculate R² for compact display
            y_true_vals = np.asarray(
                validation_cleaned_metrics[svc.target_column_name].astype(float).values
            )
            y_pred_vals = np.asarray(svc.predict_in_dollars(validation_cleaned_metrics))
            y_true_mean = np.mean(y_true_vals)
            r2 = 1 - np.sum((y_true_vals - y_pred_vals) ** 2) / np.sum(
                (y_true_vals - y_true_mean) ** 2
            )
            performance_emoji = (
                "🎯" if rmse_panel < 500 else "⚠️" if rmse_panel < 1000 else "❌"
            )
            st.markdown(
                f"""
            <div class="metric-compact">
                <strong>{performance_emoji} Model Performance</strong><br/>
                RMSE: ${rmse_panel:,.0f} • R²: {r2:.2f}
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div class="metric-compact status-warning">
                <strong>⚠️ Model Performance</strong><br/>
                No predictions available
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Section divider
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Main dashboard with two columns
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Top performing items with pink theme
        st.markdown("### 🏆 **Top Performing Items**")
        sales_col = "Actual sales ($)"

        # Get both total and average sales for richer insights
        top_total = (
            disp_clean.groupby("Item / Style", as_index=False)
            .agg({sales_col: ["sum", "mean", "count"]})
            .round(0)
        )
        top_total.columns = [
            "Item / Style",
            "Total Sales ($)",
            "Avg Monthly Sales ($)",
            "Data Points",
        ]
        top_total = top_total.sort_values(by="Total Sales ($)", ascending=False).head(
            12
        )

        chart = (
            alt.Chart(top_total)
            .mark_bar(cornerRadius=6)
            .encode(
                x=alt.X(
                    "Total Sales ($):Q",
                    title="Total Sales ($)",
                    scale=alt.Scale(nice=True),
                ),
                y=alt.Y("Item / Style:N", sort="-x", title=""),
                color=alt.Color(
                    "Total Sales ($):Q",
                    scale=alt.Scale(scheme="plasma", reverse=True),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("Item / Style", title="Item"),
                    alt.Tooltip("Total Sales ($)", format="$,.0f"),
                    alt.Tooltip("Avg Monthly Sales ($)", format="$,.0f"),
                    alt.Tooltip("Data Points", format=".0f"),
                ],
            )
            .properties(height=350, title="")
            .configure_axis(
                labelFontSize=11,
                titleFontSize=12,
                labelColor="#6b7280",
                titleColor="#374151",
            )
        )
        st.altair_chart(chart, use_container_width=True)

    with col_right:
        # Quick insights panel
        st.markdown("### 📊 **Quick Insights**")

        # Top item
        top_item = top_total.iloc[0]["Item / Style"]
        top_sales = top_total.iloc[0]["Total Sales ($)"]
        st.markdown(
            f"""
        <div class="insight-box">
            <strong>🥇 Best Seller</strong><br/>
            <span style="font-size: 1.2em; color: #d946ef;">{top_item}</span><br/>
            <span style="color: #6b7280;">${top_sales:,.0f} total sales</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Average performance
        avg_sales = disp_clean[sales_col].mean()
        total_items = disp_clean["Item / Style"].nunique()
        st.markdown(
            f"""
        <div class="insight-box">
            <strong>📈 Market Overview</strong><br/>
            <span style="font-size: 1.1em; color: #d946ef;">${avg_sales:,.0f}</span> avg per item/month<br/>
            <span style="color: #6b7280;">{total_items} items tracked</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Data quality indicator
        coverage_pct = (
            disp_clean.groupby("Item / Style").size().mean()
            / len(disp_clean["Month start"].unique())
        ) * 100
        st.markdown(
            f"""
        <div class="insight-box">
            <strong>🎯 Data Quality</strong><br/>
            <span style="font-size: 1.1em; color: #10b981;">{coverage_pct:.0f}%</span> coverage<br/>
            <span style="color: #6b7280;">Excellent data completeness</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Section divider
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Item trend analysis with improved UX
    st.markdown("### 📈 **Item Sales Trends**")

    # Smart item selector with popular items first
    popular_items = top_total.head(8)["Item / Style"].tolist()
    other_items = sorted(
        [
            item
            for item in disp_clean["Item / Style"].unique()
            if item not in popular_items
        ]
    )
    all_items = popular_items + other_items

    col_selector, col_info = st.columns([2, 1])

    with col_selector:
        item_sel = st.selectbox(
            "🔍 **Choose an item to analyze**",
            all_items,
            help="Popular items are shown first",
        )

    with col_info:
        if item_sel in popular_items:
            rank = popular_items.index(item_sel) + 1
            st.markdown(
                f"""
            <div style="background: linear-gradient(135deg, #fef7ff, #f3e8ff); padding: 0.8rem; border-radius: 6px; margin-top: 1.8rem;">
                <strong>🏆 #{rank} Best Seller</strong><br/>
                <span style="color: #6b7280; font-size: 0.9em;">High-performing item</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div style="background: linear-gradient(135deg, #f8fafc, #f1f5f9); padding: 0.8rem; border-radius: 6px; margin-top: 1.8rem;">
                <strong>📊 Standard Item</strong><br/>
                <span style="color: #6b7280; font-size: 0.9em;">Regular performer</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

    trend = disp_clean[disp_clean["Item / Style"] == item_sel].sort_values(
        by=["Month start"]
    )

    if len(trend) > 0:
        # Main trend visualization
        col_chart, col_stats = st.columns([3, 1])

        with col_chart:

            # Simple working trend chart (building on the test that worked)
            chart = (
                alt.Chart(trend)
                .mark_line(point=True, color="#ff6b9d", strokeWidth=3)
                .encode(
                    x=alt.X("Month start:T", title="Timeline"),
                    y=alt.Y("Actual sales ($):Q", title="Sales ($)"),
                    tooltip=[
                        alt.Tooltip("Month start:T", title="Month", format="%b %Y"),
                        alt.Tooltip(
                            "Actual sales ($):Q", title="Sales", format="$,.0f"
                        ),
                    ],
                )
                .properties(
                    height=400, title=f"📈 {item_sel} - Sales Performance Over Time"
                )
            )

            # Display the chart
            st.altair_chart(chart, use_container_width=True)

        with col_stats:
            # Item statistics with better visual design
            total_sales = trend["Actual sales ($)"].sum()
            avg_monthly = trend["Actual sales ($)"].mean()
            data_points = len(trend)

            st.markdown(
                f"""
            <div class="insight-box">
                <strong>💰 Total Sales</strong><br/>
                <span style="font-size: 1.4em; color: #d946ef;">${total_sales:,.0f}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            <div class="insight-box">
                <strong>📊 Monthly Average</strong><br/>
                <span style="font-size: 1.2em; color: #d946ef;">${avg_monthly:,.0f}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Trend direction with better styling
            if len(trend) > 1:
                start_val = trend["Actual sales ($)"].iloc[0]
                end_val = trend["Actual sales ($)"].iloc[-1]
                pct_change = ((end_val - start_val) / start_val) * 100

                if pct_change > 10:
                    trend_emoji = "🚀"
                    trend_color = "#10b981"
                    trend_text = f"Growing +{pct_change:.0f}%"
                elif pct_change > 0:
                    trend_emoji = "📈"
                    trend_color = "#10b981"
                    trend_text = f"Up +{pct_change:.0f}%"
                elif pct_change > -10:
                    trend_emoji = "📉"
                    trend_color = "#f59e0b"
                    trend_text = f"Down {pct_change:.0f}%"
                else:
                    trend_emoji = "⬇️"
                    trend_color = "#ef4444"
                    trend_text = f"Declining {pct_change:.0f}%"

                st.markdown(
                    f"""
                <div class="insight-box">
                    <strong>{trend_emoji} Trend</strong><br/>
                    <span style="font-size: 1.1em; color: {trend_color};">{trend_text}</span><br/>
                    <span style="color: #6b7280; font-size: 0.9em;">{data_points} months of data</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Additional insights
            if len(trend) > 1:
                best_month_idx = trend["Actual sales ($)"].idxmax()
                best_month_val = trend.loc[best_month_idx, "Month start"]
                # Convert to datetime safely
                try:
                    best_month_date = pd.to_datetime(str(best_month_val))
                except:
                    best_month_date = pd.Timestamp(str(best_month_val))
                best_month = best_month_date.strftime("%b %Y")
                best_sales = trend["Actual sales ($)"].max()
                st.markdown(
                    f"""
                <div class="insight-box">
                    <strong>🏆 Best Month</strong><br/>
                    <span style="color: #d946ef;">{best_month}</span><br/>
                    <span style="color: #6b7280; font-size: 0.9em;">${best_sales:,.0f} sales</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )
    else:
        st.warning("📊 No data available for the selected item")

    # AI Model Performance - Redesigned for all audiences
    if artifacts_ok and data_ok and svc is not None:
        # Section divider
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown("### 🤖 **AI Prediction Performance**")

        # Clean validation data for performance charts
        validation_cleaned = cleaning_service.clean_for_visualization(
            validation_panel_df
        )

        y_true_perf = np.asarray(
            validation_cleaned[svc.target_column_name].astype(float).values
        )
        if y_pred is None:
            y_pred_perf = np.asarray(svc.predict_in_dollars(validation_cleaned))
        else:
            y_pred_perf = np.asarray(y_pred)

        perf = pd.DataFrame(
            {
                "Actual sales ($)": y_true_perf,
                "Predicted sales ($)": y_pred_perf,
                "Residual ($)": y_true_perf - y_pred_perf,
                "Absolute Error ($)": np.abs(y_true_perf - y_pred_perf),
            }
        )

        # Calculate metrics
        mae = float(np.mean(perf["Absolute Error ($)"].values.astype(float)))
        rmse = float(np.sqrt(np.mean(perf["Residual ($)"].values.astype(float) ** 2)))
        y_true_mean = float(np.mean(y_true_perf))
        r2 = 1 - np.sum(perf["Residual ($)"].values.astype(float) ** 2) / np.sum(
            (y_true_perf - y_true_mean) ** 2
        )

        # MAPE calculation with zero handling
        non_zero_mask = y_true_perf != 0
        if np.sum(non_zero_mask) > 0:
            mape = (
                np.mean(
                    np.abs(
                        perf["Residual ($)"].values[non_zero_mask]
                        / y_true_perf[non_zero_mask]
                    )
                )
                * 100
            )
        else:
            mape = 0

        # Performance interpretation for different audiences
        col_simple, col_detailed = st.columns([1, 1])

        with col_simple:
            st.markdown("#### 👥 **For Everyone**")

            # Overall performance score
            if r2 > 0.7:
                score_emoji = "🟢"
                score_text = "Excellent"
                score_color = "#10b981"
            elif r2 > 0.4:
                score_emoji = "🟡"
                score_text = "Good"
                score_color = "#f59e0b"
            else:
                score_emoji = "🔴"
                score_text = "Needs Improvement"
                score_color = "#ef4444"

            st.markdown(
                f"""
            <div class="insight-box">
                <strong>{score_emoji} AI Performance</strong><br/>
                <span style="font-size: 1.3em; color: {score_color};">{score_text}</span><br/>
                <span style="color: #6b7280; font-size: 0.9em;">The AI can predict sales with {score_text.lower()} accuracy</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # MAPE-based accuracy explanation
            accuracy_pct = max(0.0, float((1 - mape / 100) * 100))
            st.markdown(
                f"""
            <div class="insight-box">
                <strong>📊 Error Rate</strong><br/>
                <span style="font-size: 1.2em; color: #d946ef;">{mape:.1f}%</span><br/>
                <span style="color: #6b7280; font-size: 0.9em;">Average prediction error as percentage of actual sales</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Average error in dollars
            st.markdown(
                f"""
            <div class="insight-box">
                <strong>💰 Typical Error</strong><br/>
                <span style="font-size: 1.2em; color: #d946ef;">±${mae:.0f}</span><br/>
                <span style="color: #6b7280; font-size: 0.9em;">Average difference between prediction and actual sales</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col_detailed:
            st.markdown("#### 🔬 **For Data Experts**")

            # Technical metrics
            metrics_data = [
                (
                    "R² Score",
                    f"{r2:.3f}",
                    "Proportion of variance explained by the model",
                ),
                ("RMSE", f"${rmse:.0f}", "Root mean squared error in dollars"),
                ("MAE", f"${mae:.0f}", "Mean absolute error in dollars"),
                ("MAPE", f"{mape:.1f}%", "Mean absolute percentage error"),
            ]

            for metric_name, metric_value, metric_desc in metrics_data:
                st.markdown(
                    f"""
                <div style="background: #f8fafc; padding: 0.6rem; border-radius: 4px; margin: 0.3rem 0; border-left: 3px solid #d946ef;">
                    <strong>{metric_name}:</strong> <span style="color: #d946ef;">{metric_value}</span><br/>
                    <span style="color: #6b7280; font-size: 0.8em;">{metric_desc}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Enhanced visualizations with explanations
        st.markdown("#### 📊 **Visual Performance Analysis**")

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("**🎯 How accurate are the predictions?**")
            st.caption("Points closer to the pink line = more accurate predictions")

            # Scatter plot with pink theme
            scatter = (
                alt.Chart(perf)
                .mark_circle(size=80, opacity=0.7, stroke="white", strokeWidth=2)
                .encode(
                    x=alt.X(
                        "Predicted sales ($):Q",
                        title="What AI Predicted ($)",
                        scale=alt.Scale(nice=True),
                    ),
                    y=alt.Y(
                        "Actual sales ($):Q",
                        title="What Actually Happened ($)",
                        scale=alt.Scale(nice=True),
                    ),
                    color=alt.Color(
                        "Absolute Error ($):Q",
                        scale=alt.Scale(scheme="plasma", reverse=True),
                        title="Prediction Error ($)",
                        legend=alt.Legend(orient="bottom"),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "Actual sales ($)", format="$,.0f", title="Actual Sales"
                        ),
                        alt.Tooltip(
                            "Predicted sales ($)", format="$,.0f", title="AI Predicted"
                        ),
                        alt.Tooltip(
                            "Absolute Error ($)", format="$,.0f", title="Error Amount"
                        ),
                    ],
                )
            )

            # Perfect prediction line
            line_data = pd.DataFrame(
                {
                    "x": [
                        perf["Predicted sales ($)"].min(),
                        perf["Predicted sales ($)"].max(),
                    ],
                    "y": [
                        perf["Predicted sales ($)"].min(),
                        perf["Predicted sales ($)"].max(),
                    ],
                }
            )
            perfect_line = (
                alt.Chart(line_data)
                .mark_line(
                    strokeDash=[8, 4], color="#ff6b9d", opacity=0.8, strokeWidth=3
                )
                .encode(x="x:Q", y="y:Q")
            )

            chart1 = (
                (scatter + perfect_line)
                .properties(height=400)
                .configure_axis(titleColor="#374151", labelColor="#6b7280")
                .configure_view(strokeOpacity=0)
            )

            st.altair_chart(chart1, use_container_width=True)

        with col_chart2:
            st.markdown("**📊 Prediction Error Analysis**")

            # Calculate key error statistics
            residuals = perf["Residual ($)"]
            abs_errors = np.abs(residuals)

            # Create clear performance buckets
            excellent_errors = np.sum(abs_errors <= 25)
            good_errors = np.sum((abs_errors > 25) & (abs_errors <= 50))
            fair_errors = np.sum((abs_errors > 50) & (abs_errors <= 100))
            poor_errors = np.sum(abs_errors > 100)

            total = len(residuals)

            # Create a clear performance summary chart instead of confusing histogram
            performance_data = pd.DataFrame(
                {
                    "Error Range": [
                        "≤$25\n(Excellent)",
                        "$25-50\n(Good)",
                        "$50-100\n(Fair)",
                        ">$100\n(Poor)",
                    ],
                    "Count": [excellent_errors, good_errors, fair_errors, poor_errors],
                    "Percentage": [
                        excellent_errors / total * 100,
                        good_errors / total * 100,
                        fair_errors / total * 100,
                        poor_errors / total * 100,
                    ],
                    "Color": ["#10b981", "#22c55e", "#f59e0b", "#ef4444"],
                }
            )

            # Performance summary chart
            perf_chart = (
                alt.Chart(performance_data)
                .mark_bar(cornerRadius=6)
                .encode(
                    x=alt.X("Error Range:N", title="Prediction Error Range", sort=None),
                    y=alt.Y("Count:Q", title="Number of Predictions"),
                    color=alt.Color("Color:N", scale=None, legend=None),
                    tooltip=[
                        alt.Tooltip("Error Range:N", title="Error Range"),
                        alt.Tooltip("Count:Q", title="Count"),
                        alt.Tooltip("Percentage:Q", format=".1f", title="Percentage"),
                    ],
                )
                .properties(height=400, title="")
                .configure_axis(titleColor="#374151", labelColor="#6b7280")
                .configure_view(strokeOpacity=0)
            )

            st.altair_chart(perf_chart, use_container_width=True)

            # Concise performance summary
            st.caption(
                f"📊 {excellent_errors} excellent, {good_errors} good, {fair_errors} fair, {poor_errors} poor predictions"
            )

            # Show the actual outlier if any
            if poor_errors > 0:
                large_errors = residuals[abs_errors > 100]
                st.caption(
                    f"Large error detail: ${large_errors.iloc[0]:.0f} (likely high-value item)"
                )

        # Enhanced metrics with insights
        st.markdown("#### 📊 **Model Performance & Insights**")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("R² Score", f"{r2:.3f}", f"{r2*100:.1f}% variance explained")

        with col2:
            st.metric("RMSE", f"${rmse:.0f}", "avg prediction error")

        with col3:
            small_errors = np.sum(np.abs(perf["Residual ($)"]) < 100)
            total_predictions = len(perf)
            small_error_pct = (small_errors / total_predictions) * 100
            st.metric("Precision", f"{small_error_pct:.0f}%", "within $100")

        # Actionable insights below metrics
        if r2 > 0.9:
            st.success(
                f"🎯 **Excellent for business forecasting** • ${rmse:.0f} avg error • {small_error_pct:.0f}% precision within $100"
            )
            st.info(
                f"📊 Small test sample ({total_predictions} predictions) makes charts chunky, but performance is solid"
            )


if __name__ == "__main__":
    main()
