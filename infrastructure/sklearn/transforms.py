"""Custom transformers for the forecasting pipeline."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class LogInverseTransformRegressor(BaseEstimator, RegressorMixin):
    """Wrapper that applies expm1 to model predictions to convert from log space to dollars."""

    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        # This wrapper doesn't need fitting, the inner regressor is already fitted
        return self

    def predict(self, X):
        log_pred = self.regressor.predict(X)
        return np.expm1(log_pred)
