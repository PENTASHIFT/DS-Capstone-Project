from sklearn.metrics import mean_absolute_error
import xgboost as xgb

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    make_scorer,
)


def train_xgboost_model(
    df,
    target_column,
    columns_to_drop,
    test_size=0.2,
    random_state=42,
    print_results=True,
    n_jobs=1,
):
    """
    Trains an XGBoost model on the provided DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        columns_to_drop (list): List of column names to drop from the features.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.
        print_results (bool): Whether to print evaluation metrics.
        n_jobs (int): Number of parallel jobs to run. If -1, use all processors.

    Returns:
        dict: A dictionary containing the trained model, feature importances, and evaluation metrics.
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

    ### Model training and evaluation
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
        n_jobs=n_jobs,
    )

    xgb_regressor.fit(X_train, y_train)

    y_pred = xgb_regressor.predict(X_test)

    ### Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    ### Feature importances
    importances = xgb_regressor.feature_importances_
    feature_names = X.columns
    sorted_indices = np.argsort(importances)[::-1]

    if print_results:
        print(f"Model trained for target: {target_column}")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R^2 Score: {r2:.4f}")

    return {
        "model": xgb_regressor,
        "y_pred": y_pred,
        "y_test": y_test,
        "target_column": target_column,
        "metrics": {"MSE": mse, "RMSE": rmse, "MAE": mae, "R^2": r2},
    }


def cross_validate_xgboost(
    df,
    target_column,
    columns_to_drop,
    test_size=0.2,
    random_state=42,
    print_results=True,
    n_jobs=1,
    xgb_params=None,
    n_splits=5,
):
    """
    Performs k-fold cross-validation on an XGBoost model.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        columns_to_drop (list): List of column names to drop from the features.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.
        print_results (bool): Whether to print cross-validation results.
        n_jobs (int): Number of parallel jobs to run. If -1, use all processors.
        xgb_params (dict): Optional parameters for XGBoost model. If None, default parameters are used.
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

    ### Define default XGBoost parameters
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "missing": np.nan,
            "random_state": random_state,
        }

    ### Model training and evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    xgb_regressor = xgb.XGBRegressor(**xgb_params)
    xgb_regressor.fit(X_train, y_train)
    y_pred = xgb_regressor.predict(X_test)

    ### Cross-validation setup
    rmse_scorer = make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=False,
    )
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    r2_scorer = make_scorer(r2_score, greater_is_better=True)

    ### K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rmse_scores = -cross_val_score(
        xgb_regressor, X, y, cv=kf, scoring=rmse_scorer, n_jobs=n_jobs
    )
    mae_scores = -cross_val_score(
        xgb_regressor, X, y, cv=kf, scoring=mae_scorer, n_jobs=-n_jobs
    )
    r2_scores = cross_val_score(
        xgb_regressor, X, y, cv=kf, scoring=r2_scorer, n_jobs=-n_jobs
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
        "R^2": {"mean": r2_scores.mean(), "std": r2_scores.std(), "scores": r2_scores},
    }

    ### Feature importances
    importances = xgb_regressor.feature_importances_
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
        print(f"R^2: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

    ### Fit the model on the entire dataset
    xgb_regressor.fit(X, y)

    return {
        "model": xgb_regressor,
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
