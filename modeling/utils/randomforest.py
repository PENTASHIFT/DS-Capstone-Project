from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def train_random_forest_model(
    df, target_column, columns_to_drop, test_size=0.2, random_state=42, print_plot=True
):
    """
    Trains a Random Forest regression model and evaluates its performance.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        columns_to_drop (list): List of column names to drop from the features.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing the trained model, predictions, and evaluation metrics.
    """
    # Drop specified columns
    X = df.drop(columns=columns_to_drop)
    X = X.select_dtypes(include=[np.number])  # Select only numeric columns
    y = df[target_column]

    # Align X and y by dropping rows where y is NaN
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=random_state)

    # Fit the model
    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Get feature importances
    importances = rf_regressor.feature_importances_
    feature_names = X.columns
    sorted_indices = np.argsort(importances)[::-1]

    if print_plot:
        top_n = 10
        top_indices = sorted_indices[:top_n]
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n)[::-1], importances[top_indices], align="center")
        plt.yticks(range(top_n)[::-1], feature_names[top_indices])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Random Forest Feature Importances: {target_column}")
        plt.tight_layout()
        plt.show()

    return {
        "model": rf_regressor,
        "y_pred": y_pred,
        "y_test": y_test,
        "metrics": {"MSE": mse, "RMSE": rmse, "R²": r2},
    }


def plot_rf_residuals(y_test, y_pred, target_column):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color="blue")
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"Random Forest Residual Plot: {target_column}")
    plt.show()


def plot_rf_actual_predicted(y_test, y_pred, target_column):
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
    plt.title(f"Random Forest: Actual vs Predicted for {target_column}")
    plt.show()
