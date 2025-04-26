from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_lasso_model(
    df: pd.DataFrame, target_column: str, columns_to_drop: list[str]
) -> list[tuple[str, float]]:
    """
    Fits a Lasso regression model on the data and returns sorted feature coefficients
    Drops all columsn with the given prefix

    Args:
        df: pandas DataFrame containing features and target
        target_column: name of the target column
        columns_to_drop (list): List of column names to drop from the features.

    Returns:
        sorted list of (feature, coefficient) pairs
    """

    # Drop specified columns
    X = df.drop(columns=columns_to_drop)
    y = df[target_column]

    # Drop rows where y is NaN
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    # Select only numeric columns for the pipeline
    X_numeric = X.select_dtypes(include=[np.number])

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=5, random_state=42, max_iter=10000)),
        ]
    )

    # Fit the model
    pipe.fit(X_numeric, y)

    lasso_cv = pipe.named_steps["lasso"]

    # Extract coefficients after fitting
    lasso_coefficients = lasso_cv.coef_
    coef_feature_pairs = list(zip(X_numeric.columns, lasso_coefficients))
    sorted_pairs = sorted(coef_feature_pairs, key=lambda x: abs(x[1]), reverse=True)

    return X_numeric, y, sorted_pairs


def plot_lasso_coefficients(
    target_column: str, sorted_pairs: list[tuple[str, float]]
) -> None:
    """Plots the top 10 feature importances from Lasso regression using matplotlib
    and seaborn.
    This function takes the target column name and a list of sorted pairs (feature, coefficient)
    and creates a horizontal bar plot of the top 10 features with their corresponding coefficients.


    Args:
        target_column (str): String representing the target column name.
        sorted_pairs (list[tuple[str, float]]): List of tuples containing feature names and their coefficients.

    Returns:
        None
    """
    top_n = 10
    top_features = sorted_pairs[:top_n]

    plt.figure(figsize=(10, 6))
    plt.barh([f[0] for f in top_features], [f[1] for f in top_features])
    plt.xlabel("Coefficient Value")
    plt.title(f"Top 10 Feature Importances from Lasso Regression: {target_column}")
    plt.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.show()


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer


def train_and_evaluate_lasso_model(
    X: np.ndarray,
    y: np.ndarray,
    model: LassoCV,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Trains and evaluates a Lasso regression model on the provided data.

    Args:
        X: Features dataframe/array
        y: Target variable
        model: Initialized model object
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        dict: Dictionary containing trained model and evaluation metrics
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Handle missing values in X_train and X_test
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r_2 = model.score(X_test, y_test)

    return {
        "model": model,
        "y_pred": y_pred,
        "y_test": y_test,
        "mse": mse,
        "rmse": rmse,
        "r2": r_2,
    }


def plot_actual_predicted_lasso(
    y_test: pd.Series, y_pred: np.ndarray, target_column: str
) -> None:
    """
    Plots the actual vs. predicted values for a Lasso regression model.

    Args:
        y_test (pd.Series): The actual target values from the test set.
        y_pred (np.ndarray): The predicted target values from the model.
        target_column (str): The name of the target column.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Lasso Regression: Actual vs Predicted for {target_column}")
    plt.show()


def plot_lasso_residuals(
    y_test: pd.Series, y_pred: np.ndarray, target_column: str
) -> None:
    """
    Plots the residuals for a Lasso regression model.

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
    plt.title(f"Lasso Regression Residual Plot: {target_column}")
    plt.show()
