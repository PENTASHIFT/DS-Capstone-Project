from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    make_scorer,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def train_random_forest_model(
    df,
    target_column,
    columns_to_drop,
    test_size=0.2,
    random_state=42,
    print_results=True,
    n_jobs=1,
):
    """
    Trains a Random Forest regression model and evaluates its performance.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        columns_to_drop (list): List of column names to drop from the features.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.
        print_results (bool): Whether to print model evaluation results.
        n_jobs (int): Number of jobs to run in parallel.

    Returns:
        dict: A dictionary containing the trained model, predictions, and evaluation metrics.
    """
    ### Data preparation
    X = df.drop(columns=columns_to_drop)
    X = X.select_dtypes(include=[np.number])
    y = df[target_column]

    ### Align X and y by dropping rows where y is NaN or infinite
    y = pd.to_numeric(df[target_column], errors="coerce")
    mask = y.notna() & np.isfinite(y)
    X = X.loc[mask]
    y = y.loc[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    rf_regressor = RandomForestRegressor(
        n_estimators=100, random_state=random_state, n_jobs=n_jobs
    )

    ### Fit the model
    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))

    ### Feature importances
    importances = rf_regressor.feature_importances_
    feature_names = X.columns

    if print_results:
        print(f"Model trained for target: {target_column}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R^2 Score: {r2:.4f}")

    return {
        "model": rf_regressor,
        "X": X,
        "y_pred": y_pred,
        "y_test": y_test,
        "metrics": {"MSE": mse, "RMSE": rmse, "R²": r2},
        "feature_importances": dict(zip(feature_names, importances)),
        "feature_names": feature_names,
    }


def cross_validate_random_forest(
    df,
    target_column,
    columns_to_drop,
    test_size=0.2,
    random_state=42,
    print_results=True,
    n_jobs=1,
    rf_params=None,
    n_splits=5,
):
    """
    Performs k-fold cross-validation on a Random Forest regression model.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        columns_to_drop (list): List of column names to drop from the features.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.
        print_results (bool): Whether to print cross-validation results.
        n_jobs (int): Number of jobs to run in parallel.
        rf_params (dict): Optional parameters for Random Forest model. If None, default parameters are used.
        n_splits (int): Number of folds for cross-validation.

    Returns:
        dict: A dictionary containing cross-validation scores and the final trained model.
    """

    ### Data preparation
    X = df.drop(columns=columns_to_drop)
    X = X.select_dtypes(include=[np.number])
    y = df[target_column]

    ### Align X and y by dropping rows where y is NaN or infinite
    y = pd.to_numeric(df[target_column], errors="coerce")
    mask = y.notna() & np.isfinite(y)
    X = X.loc[mask]
    y = y.loc[mask]

    ### Define default Random Forest parameters
    if rf_params is None:
        rf_params = {
            "n_estimators": 100,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }

    ### Model training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    rf_regressor = RandomForestRegressor(
        n_estimators=100, random_state=random_state, n_jobs=n_jobs
    )
    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(X_test)

    ### Cross-validation setup
    rmse_scorer = make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=False,
    )
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    r2_scorer = make_scorer(r2_score, greater_is_better=True)

    ### K-Fold Cross-Validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rmse_scores = -cross_val_score(
        rf_regressor, X, y, cv=kf, scoring=rmse_scorer, n_jobs=n_jobs
    )
    mae_scores = -cross_val_score(
        rf_regressor, X, y, cv=kf, scoring=mae_scorer, n_jobs=-n_jobs
    )
    r2_scores = cross_val_score(
        rf_regressor, X, y, cv=kf, scoring=r2_scorer, n_jobs=-n_jobs
    )

    ### Cross-validation results
    cv_results = {
        "RMSE": {
            "mean": rmse_scores.mean(),
            "std": rmse_scores.std(),
            "scores": rmse_scores,
        },
        "MAE": {
            "mean": mae_scores.mean(),
            "std": mae_scores.std(),
            "scores": mae_scores,
        },
        "MSE": {
            "mean": (rmse_scores**2).mean(),
            "std": (rmse_scores**2).std(),
            "scores": rmse_scores**2,
        },
        "R^2": {"mean": r2_scores.mean(), "std": r2_scores.std(), "scores": r2_scores},
    }

    ### Feature importances
    importances = rf_regressor.feature_importances_
    feature_names = X.columns

    if print_results:
        if "Year" in df.columns:
            year = df["Year"].iloc[0]
        else:
            year = df["AcademicYear"].iloc[0]
        print(
            f"Cross-Validation Results for {target_column} ({n_splits} folds) for {year}:"
        )
        print(f"RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
        print(f"MAE: {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
        print(f"R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

    ### Fit the model on the entire dataset
    rf_regressor.fit(X, y)

    return {
        "model": rf_regressor,
        "X": X,
        "y_pred": y_pred,
        "y_test": y_test,
        "metrics": {
            "MSE": cv_results["RMSE"],
            "RMSE": cv_results["RMSE"],
            "R^2": cv_results["R^2"],
            "MAE": cv_results["MAE"],
        },
        "feature_importances": dict(zip(feature_names, importances)),
        "feature_names": feature_names,
    }
