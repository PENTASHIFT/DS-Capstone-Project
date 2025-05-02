from sklearn.metrics import mean_absolute_error
import xgboost as xgb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def train_xgboost_model(
    df, target_column, columns_to_drop, test_size=0.2, random_state=42
):
    """
    Trains an XGBoost model on the provided DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        columns_to_drop (list): List of column names to drop from the features.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing the trained model, feature importances, and evaluation metrics.
    """
    X = df.drop(columns=columns_to_drop)

    # Select only numeric columns
    X = X.select_dtypes(include=[np.number])
    y = df[target_column]

    # Align X and y by dropping rows where y is NaN or infinite
    mask = y.notna() & np.isfinite(y)
    X = X.loc[mask]
    y = y.loc[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    xgb_regressor = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        missing=np.nan,
        random_state=random_state,
    )

    # Fit the model
    xgb_regressor.fit(X_train, y_train)

    y_pred = xgb_regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Get feature importances
    importances = xgb_regressor.feature_importances_
    feature_names = X.columns
    sorted_indices = np.argsort(importances)[::-1]

    # Plot feature importances
    top_n = 10
    top_indices = sorted_indices[:top_n]
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n)[::-1], importances[top_indices], align="center")
    plt.yticks(range(top_n)[::-1], feature_names[top_indices])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"XGBoost Feature Importances: {target_column}")
    plt.tight_layout()
    plt.show()

    return {
        "model": xgb_regressor,
        "y_pred": y_pred,
        "y_test": y_test,
        "target_column": target_column,
        "metrics": {"MSE": mse, "RMSE": rmse, "MAE": mae, "R^2": r2},
    }


def plot_xgb_residuals(
    y_test: pd.Series, y_pred: np.ndarray, target_column: str
) -> None:
    """
    Plots the residuals for an XGBoost regression model.

    Args:
        y_test (pd.Series): The actual target values from the test set.
        y_pred (np.ndarray): The predicted target values from the model.
        target_column (str): The name of the target column.

    Returns:
        None
    """

    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color="blue")
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"XGBoost Regression Residual Plot: {target_column}")
    plt.show()


def plot_xgb_predicted_actual(
    y_test: pd.Series, y_pred: np.ndarray, target_column: str
) -> None:
    """
    Plots the actual vs. predicted values for an XGBoost regression model.

    Args:
        y_test (pd.Series): The actual target values from the test set.
        y_pred (np.ndarray): The predicted target values from the model.
        target_column (str): The name of the target column.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red",
        linestyle="--",
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"XGBoost Regression: Actual vs Predicted for {target_column}")
    plt.show()
