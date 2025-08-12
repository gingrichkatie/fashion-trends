# Data Directory

This directory contains the data assets for the Fashion Trends Forecasting project, organized following clean architecture principles.

## Directory Structure

### `raw/`
- **`Fashion_Retail_Sales.csv`**: Original dataset used for training and analysis
  - Contains fashion retail transaction data with purchase amounts, ratings, payment methods, etc.
  - Used by `Fashion_Trends_Forecasting.ipynb` for model training
  - Loaded by `infrastructure/data/real_data_loader.py` for the Streamlit application

## Data Sources

The raw data serves as the single source of truth for training and validation. All processed datasets and model artifacts are derived from these raw files.

## Usage

- **Training**: The Jupyter notebook loads data from `raw/Fashion_Retail_Sales.csv`
- **Application**: The Streamlit app uses `RealDataLoader` to process the same raw data for consistency
- **Testing**: Tests can reference raw data for fixture generation when needed

## Data Governance

- Raw data files should be versioned and treated as immutable once committed
- Any data transformations should be reproducible through code (not manual edits)
- Processed data outputs should go to appropriate directories (e.g., `artifacts/fashion_sales_monthly_pipeline.pkl` for the trained model)
