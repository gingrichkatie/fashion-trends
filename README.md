## Fashion Trends Forecasting

Purpose: Forecast monthly sales per item for a fashion retailer and explore trends in a Streamlit app.

### What's in this repo
- **`streamlit_app.py`**: Single Streamlit UI wired to SOLID/Clean Architecture
- **`infrastructure/sklearn/`**: Model loading and prediction services
- **`core/use_cases/`**: Business logic orchestration (intervals, scale detection)
- **`domain/models.py`**: Core domain types and value objects
- **`data/raw/Fashion_Retail_Sales.csv`**: Training dataset
- **`artifacts/fashion_sales_monthly_pipeline.pkl`**: Single trained model artifact
- **`Fashion_Trends_Forecasting.ipynb`**: Baseline training notebook (exploration)
- **`enhanced_training.py`**: Production model training with advanced techniques

### Quickstart
1) Create venv and install deps
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Ensure pipeline artifact exists (produced by the notebook)
- `artifacts/fashion_sales_monthly_pipeline.pkl`

3) Run the app
```
streamlit run streamlit_app.py
```

### Training Workflow

#### 1. Notebook Training (Baseline)
- **File**: `Fashion_Trends_Forecasting.ipynb`
- **Purpose**: Initial exploration, EDA, and baseline model development
- **Process**:
  - Loads raw data from `data/raw/Fashion_Retail_Sales.csv`
  - Performs exploratory data analysis and feature engineering
  - Creates basic pipeline with HistGradientBoostingRegressor
  - Time-series validation with 80/20 train/test split
  - **Output**: Legacy format artifacts (separate model/preprocessor files)

#### 2. Enhanced Training (Production)
- **File**: `enhanced_training.py`
- **Purpose**: High-performance model training with advanced techniques
- **When to use**: After notebook exploration, for production-grade models
- **Process**:
  - Uses `RealDataLoader` for enhanced feature engineering (40+ features vs 11 baseline)
  - Tests multiple algorithms: HistGradientBoosting, RandomForest, ExtraTrees
  - Advanced preprocessing: RobustScaler, improved target encoding
  - Comprehensive evaluation: Time-series CV, multiple metrics (MAE, RMSE, R², MAPE)
  - Hyperparameter optimization available via grid search
  - **Output**: Single production pipeline at `artifacts/fashion_sales_monthly_pipeline.pkl`

#### Running Enhanced Training
```bash
# After notebook exploration is complete:
python enhanced_training.py
```

**Performance Gains**: Enhanced training typically achieves:
- **R² Score**: 0.8+ (vs baseline 0.033) - 20x+ improvement
- **RMSE**: <$200 (vs baseline $533.95) - 2.5x+ improvement
- **MAE**: <$100 (vs baseline $231.03) - 2x+ improvement

The app consumes the enhanced pipeline from `artifacts/fashion_sales_monthly_pipeline.pkl`.

#### Technical Details

**Enhanced Features** (via `RealDataLoader`):
- Advanced time-based features: rolling windows, lag variables, trend indicators
- Seasonal decomposition and cyclical patterns
- Cross-category feature interactions
- Statistical aggregations per item and time period
- Feature scaling and outlier-robust transformations

**Algorithm Comparison**:
- **HistGradientBoosting**: Fast, memory-efficient, handles missing values natively
- **RandomForest**: Robust to overfitting, good feature importance insights
- **ExtraTrees**: Extra randomization for variance reduction, often best performance

**Advanced Preprocessing**:
- `RobustScaler`: More resistant to outliers than StandardScaler
- `TargetEncoder`: Improved categorical encoding with reduced smoothing (0.1 vs 0.3)
- `TransformedTargetRegressor`: Log transformation for better handling of sales distribution
- Missing value handling optimized for time series data

**Evaluation Methodology**:
- Time series cross-validation (respects temporal order)
- Multiple metrics: MAE, RMSE, R², MAPE for comprehensive assessment
- Training vs test performance monitoring to detect overfitting
- Cross-validation stability analysis

### Architecture (DDD/Clean)
- **Domain**: `domain/models.py` defines domain types and value objects
- **Core**: `core/use_cases` holds orchestration logic (scale detection, forecasting, intervals)
- **Infrastructure**: `infrastructure/sklearn` loads and executes the sklearn model; `infrastructure/data` produces demo data
- **Presentation**: `streamlit_app.py` contains UI only, using dependency injection and caching

See `PROJECT_STRUCTURE.md` for detailed architecture documentation.

### Performance
- Model and preprocessors are cached via `st.cache_resource`.
- Computation-heavy predictions and panel generation are cached via `st.cache_data`.

### Testing
- Create tests under `tests/` using pytest with the Arrange–Act–Assert pattern.
- Avoid mocking domain models; generate small, real `pandas` frames as fixtures inline per test.

### Maintenance
- Keep UI dumb; evolve business logic in application/infrastructure layers.
- If artifact schema changes, update `model_repository.py` and `forecasting_service.py` accordingly.
