from sklearn.metrics import mean_absolute_error
import xgboost as xgb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    make_scorer
)


def train_xgboost_model(
    df,
    target_column,
    columns_to_drop,
    test_size=0.2,
    random_state=42,
    print_plot=True,
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

    Returns:
        dict: A dictionary containing the trained model, feature importances, and evaluation metrics.
    """
    X = df.drop(columns=columns_to_drop)

    # Select only numeric columns
    X = X.select_dtypes(include=[np.number])
    y = df[target_column]

    # Align X and y by dropping rows where y is NaN or infinite
    y = pd.to_numeric(df[target_column], errors="coerce")
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
        n_jobs=n_jobs,
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
    if print_plot:
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


plt.style.use("default")


def plot_top_k_features(models, model_type, importance_type, target_variable: str, k=5):
    feature_importances = {}

    for year, model in models.items():
        importance = (
            model["model"].get_booster().get_score(importance_type=importance_type)
        )
        sorted_importance = sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        )[:k]
        feature_importances[year] = {
            feature: score for feature, score in sorted_importance
        }

    df = pd.DataFrame(feature_importances).T.fillna(0)
    mean_importance = df.mean().sort_values(ascending=False)
    top_features = mean_importance.head(k).index.tolist()
    df_top = df[top_features]

    plt.figure(figsize=(14, 6))
    ax = df_top.plot(kind="bar", figsize=(14, 6), width=0.8)

    plt.title(
        f"Top {k} Features by Year for {target_variable} ({model_type}, {importance_type})"
    )
    plt.ylabel("Feature Importance")
    plt.xlabel("Year")
    plt.xticks(rotation=45)
    plt.legend(title="Feature", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # Return list of top k features across all years
    all_features = []
    for year, features in feature_importances.items():
        all_features.append(pd.Series(features, name=year))

    all_features_df = pd.concat(all_features, axis=1)
    top_features = all_features_df.T.mean(axis=0)
    top_features = top_features.sort_values(ascending=False).head(k)
    return top_features.index.tolist()


def plot_feature_avg_variance(
    models, model_type, importance_type, target_variable, k=5
):
    feature_importances = []

    for year, model in models.items():
        importance = (
            model["model"].get_booster().get_score(importance_type=importance_type)
        )
        feature_importances.append(pd.Series(importance))

    df = pd.DataFrame(feature_importances).fillna(0)
    avg_importance = df.mean(axis=0)
    var_importance = df.var(axis=0)

    top_features = avg_importance.sort_values(ascending=False).head(k).index
    avg_top = avg_importance[top_features]
    var_top = var_importance[top_features]

    plt.figure(figsize=(14, 6))

    ax = plt.gca()
    ax.bar(
        range(len(top_features)),
        avg_top,
        yerr=var_top,
        capsize=4,
        color="skyblue",
        edgecolor="black",
        alpha=0.8,
        label="Average Importance",
        zorder=2,
    )

    ax.grid(True, linestyle="--", alpha=0.3)

    plt.title(
        f"Top {k} Average Feature Importances with Variance for {target_variable} ({model_type}, {importance_type})"
    )
    plt.ylabel("Average Feature Importance")
    plt.xlabel("Features")
    plt.xticks(range(len(top_features)), top_features, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_time_series(
    models, model_type, importance_type, target_variable, features
):
    feature_importances = {}

    for year, model_dict in models.items():
        importance = model_dict["model"].get_booster().get_score(importance_type="gain")
        feature_importances[year] = {
            feature: importance.get(feature, 0) for feature in features
        }

    df = pd.DataFrame(feature_importances).T
    df.index.name = "Year"

    fig, ax = plt.subplots(figsize=(14, 6))

    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "+"]
    marker_cycle = markers * (len(features) // len(markers) + 1)

    for i, feature in enumerate(features):
        ax.plot(df.index, df[feature], marker=marker_cycle[i], label=feature)

    ax.legend(title="Features", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(
        f"Feature Importance Over Time for {target_variable} ({model_type}, {importance_type})"
    )
    ax.set_ylabel("Feature Importance")
    ax.set_xlabel("Year")

    plt.tight_layout()
    plt.show()


def cross_validate_xgboost(
    df,
    target_column,
    columns_to_drop,
    n_splits=5,
    random_state=42,
    print_results=True,
    print_plot=False,
    xgb_params=None,
    n_jobs=1,
):
    """
    Performs k-fold cross-validation on an XGBoost model.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        columns_to_drop (list): List of column names to drop from the features.
        n_splits (int): Number of folds for cross-validation.
        random_state (int): Random seed for reproducibility.
        print_results (bool): Whether to print cross-validation results.
        xgb_params (dict): Optional parameters for XGBoost model. If None, default parameters are used.

    Returns:
        dict: A dictionary containing cross-validation scores and the final trained model.
    """
    X = df.drop(columns=columns_to_drop)
    X = X.select_dtypes(include=[np.number])  # Select only numeric columns
    y = df[target_column]

    y = pd.to_numeric(df[target_column], errors="coerce")
    mask = y.notna() & np.isfinite(y)
    X = X.loc[mask]
    y = y.loc[mask]

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

    xgb_regressor = xgb.XGBRegressor(**xgb_params)

    rmse_scorer = make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=False,
    )
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    r2_scorer = make_scorer(r2_score, greater_is_better=True)

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

    if print_plot:
        plt.figure(figsize=(12, 6))

        metrics = ["RMSE", "MAE", "R^2"]
        scores = [rmse_scores, mae_scores, r2_scores]
        colors = ["#FF9999", "#66B2FF", "#99CC99"]

        for i, (metric, score, color) in enumerate(zip(metrics, scores, colors)):
            plt.subplot(1, 3, i + 1)
            plt.boxplot(score, patch_artist=True, boxprops=dict(facecolor=color))
            plt.title(f"{metric} Cross-Validation")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

        plt.show()

    xgb_regressor.fit(X, y)

    return {
        "model": xgb_regressor,
        "cv_results": cv_results,
        "feature_names": X.columns.tolist(),
        "target_column": target_column,
        "Year": (
            df["Year"].iloc[0] if "Year" in df.columns else df["AcademicYear"].iloc[0]
        ),
    }


def plot_cv_feature_importance(cv_results, top_n=5):
    """
    Plots feature importances from a dictionary of cross-validated XGBoost models.

    Args:
        cv_results (dict): A dictionary where the key is the year and the value is the model corresponding to that year.
        top_n (int): Number of top features to display.

    Returns:
        list: A sorted list of top features across all years.
    """
    feature_importances = {}

    for year, result in cv_results.items():
        model = result["model"]
        feature_names = result["feature_names"]
        target_column = result["target_column"]

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        feature_importances[year] = {
            feature_names[i]: importances[i] for i in indices[:top_n]
        }

    df = pd.DataFrame(feature_importances).T.fillna(0)
    mean_importance = df.mean().sort_values(ascending=False)
    top_features = mean_importance.head(top_n).index.tolist()
    df_top = df[top_features]

    plt.figure(figsize=(14, 6))
    ax = df_top.plot(kind="bar", figsize=(14, 6), width=0.8)

    plt.title(f"Top {top_n} Features Across Years (Cross-Validated)")
    plt.ylabel("Feature Importance")
    plt.xlabel("Year")
    plt.xticks(rotation=45)
    plt.legend(title="Feature", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    return top_features
