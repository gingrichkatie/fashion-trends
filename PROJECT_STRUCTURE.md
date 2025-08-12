# Project Structure

This project follows **Clean Architecture** for maintainable, testable code.

## Architecture Overview

```
fashion-trends-forecasting/
├── streamlit_app.py             # Presentation Layer - Single Streamlit UI entrypoint
├── core/                        # Core Business Logic Layer
│   └── use_cases/               # Business logic orchestration
│       ├── intervals.py         # Prediction interval calculations
│       └── scale_detection.py   # Model output scale detection
├── domain/                      # Domain Layer
│   └── models.py                # Core domain types and value objects
├── infrastructure/              # Infrastructure Layer
│   ├── sklearn/                 # ML model services
│   │   ├── model_repository.py  # Model loading and persistence
│   │   └── forecasting_service.py # Prediction orchestration
│   └── data/                    # Data access
│       ├── demo_panel_generator.py # Synthetic data generation
│       └── real_data_loader.py  # Real data processing
├── data/                        # Data Assets
│   └── raw/                     # Raw training data
│       └── Fashion_Retail_Sales.csv
├── artifacts/                   # Model Artifacts
│   └── fashion_sales_monthly_pipeline.pkl
├── tests/                       # Test Suite
└── Fashion_Trends_Forecasting.ipynb # Training notebook
```

## Entry Points

- **Production**: `streamlit run streamlit_app.py`
- **Training**: `Fashion_Trends_Forecasting.ipynb`
- **Testing**: `pytest tests/`
