from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("default")

category_mapping = {
    "RB": "African American",
    "RI": "American Indian or Alaska Native",
    "RA": "Asian",
    "RF": "Filipino",
    "RH": "Hispanic or Latino",
    "RD": "Not Reported",
    "RP": "Pacific Islander",
    "RT": "Two or More Races",
    "RW": "White",
    "GM": "Male",
    "GF": "Female",
    "GX": "Non-Binary Gender",
    "GZ": "Missing Gender",
    "SE": "English Learners",
    "SD": "Students with Disabilities",
    "SS": "Socioeconomic",
    "SM": "Migrant",
    "SF": "Foster",
    "SH": "Homeless",
    "09": "9th Grade",
    "10": "10th Grade",
    "11": "11th Grade",
    "12": "12th Grade",
    "TA": "Total",
}

metric_mapping = {
    "MeritRate": "Merit Rate",
    "OtherRate": "Other Rate",
    "SchoolCode": "School Code",
    "Merit": "Merit",
    "AdultEd": "Adult Education",
    "CohortStudents": "Cohort Students",
    "BiliteracyRate": "Biliteracy Rate",
    "RegHSDiplomaRate": "Graduation Rate",
    "DropoutRate": "Dropout Rate",
    "UniReqsPercent": "College Readiness Rate",
    "StillEnrolledRate": "Still Enrolled Rate",
    "AdultEdRate": "Adult Education Rate",
    "ExemptionRate": "Exemption Rate",
    "GEDRate": "GED Rate",
    "EO": "English Only",
    "RFEP": "Reclassified Fluent English Proficient",
    "CPP": "California Proficiency Program",
    "IFEP": "Initial Fluent English Proficient",
    "EL03Y": "English Learner for 0-3 years",
    "EL6+Y": "English Learner for 6+ years",
    "CPPRate": "California Proficiency Program Rate",
    "LTEL": "Long-Term English Learner",
    "AR": "At-Risk",
}


def format_feature_name(feature):
    if "." in feature:
        metric, category = feature.split(".")
        # Get readable metric and category names from mappings if available
        if "metric_mapping" in globals() and metric in metric_mapping:
            metric = metric_mapping[metric]
        if "category_mapping" in globals() and category in category_mapping:
            category = category_mapping[category]
        return f"{metric} - {category}"
    return feature


def plot_top_features_rf(cv_results, target_variable, top_n=10, figsize=(12, 8)):
    """
    Plots the top features by importance for a Random Forest model in a horizontal bar chart.

    Args:
        cv_results (dict): Dictionary containing cross-validation results
        target_variable (str): Name of the target variable for the plot title
        top_n (int): Number of top features to display
        figsize (tuple): Figure size as (width, height)

    Returns:
        list: List of the top feature names
    """
    model = cv_results["model"]
    feature_names = cv_results["feature_names"]
    importances = dict(zip(feature_names, model.feature_importances_))

    avg_importances = {feature: value for feature, value in importances.items()}
    top_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)[
        :top_n
    ]

    top_features = top_features[::-1]

    formatted_names = [format_feature_name(f[0]) for f in top_features]
    importance_values = [f[1] for f in top_features]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = range(len(formatted_names))
    ax.barh(
        y_pos,
        importance_values,
        align="center",
        color="forestgreen",
        edgecolor="darkgreen",
        alpha=0.8,
    )

    r2_key = "R^2" if "R^2" in cv_results["cv_results"] else "R²"

    model_stats = (
        f"RMSE: {cv_results['cv_results']['RMSE']['mean']:.4f} ± {cv_results['cv_results']['RMSE']['std']:.4f}\n"
        f"MAE: {cv_results['cv_results']['MAE']['mean']:.4f} ± {cv_results['cv_results']['MAE']['std']:.4f}\n"
        f"R²: {cv_results['cv_results'][r2_key]['mean']:.4f} ± {cv_results['cv_results'][r2_key]['std']:.4f}"
    )

    ax.text(
        0.98,
        0.05,
        model_stats,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(formatted_names)
    ax.set_xlabel("Feature Importance", fontsize=12)

    formatted_target = format_feature_name(target_variable)
    ax.set_title(
        f"Top Features by Importance for {formatted_target} (Random Forest)",
        fontsize=14,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

    return [
        f[0]
        for f in sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
    ]


def train_random_forest_model(
    df,
    target_column,
    columns_to_drop,
    test_size=0.2,
    random_state=42,
    print_plot=False,
    print_results=False,
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

    rf_regressor = RandomForestRegressor(
        n_estimators=100, random_state=random_state, n_jobs=n_jobs
    )

    # Fit the model
    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))

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


def plot_top_k_features_rf(models, target_variable: str, k=5):
    """
    Plots the top k features based on average importance across all years.

    Args:
        models (dict): Dictionary of Random Forest models by year.
        target_variable (str): Name of the target variable.
        k (int): Number of top features to show.

    Returns:
        list: List of top features.
    """
    feature_importances = {}

    # Format feature names for better readability
    def format_feature_name(feature):
        if "." in feature:
            metric, category = feature.split(".")
            # Get readable metric and category names from mappings if available
            if "metric_mapping" in globals() and metric in metric_mapping:
                metric = metric_mapping[metric]
            if "category_mapping" in globals() and category in category_mapping:
                category = category_mapping[category]
            return f"{metric} - {category}"
        return feature

    for year, model_dict in models.items():
        model = model_dict["model"]
        feature_names = model_dict["X"].columns
        importances = model.feature_importances_
        feature_importances[year] = dict(zip(feature_names, importances))

    df = pd.DataFrame(feature_importances).T.fillna(0)
    mean_importance = df.mean().sort_values(ascending=False)
    top_features = mean_importance.head(k).index.tolist()
    df_top = df[top_features]

    plt.figure(figsize=(14, 6))
    ax = df_top.plot(kind="bar", figsize=(14, 6), width=0.8)

    # Get average R² if available
    r2_values = [m.get("metrics", {}).get("R²", None) for m in models.values()]
    r2_values = [r2 for r2 in r2_values if r2 is not None]
    avg_r2 = sum(r2_values) / len(r2_values) if r2_values else None

    plt.title(
        f"Top {k} Features by Year for {format_feature_name(target_variable)}"
        + (f" (Avg. R²: {avg_r2:.4f})" if avg_r2 else "")
    )
    plt.xlabel("Year")
    plt.ylabel("Feature Importance")
    plt.legend([format_feature_name(f) for f in top_features], title="Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return top_features


def plot_feature_avg_variance_rf(models, target_variable: str, top_k_features):
    """
    Plots the average importance and variance of the top k features.

    Args:
        models (dict): Dictionary of Random Forest models by year.
        top_k_features (list): List of top k features to plot.
    """
    feature_importances = {}

    for year, model_dict in models.items():
        model = model_dict["model"]
        feature_names = model_dict["X"].columns
        importances = model.feature_importances_
        feature_importances[year] = dict(zip(feature_names, importances))

    df = pd.DataFrame(feature_importances).T.fillna(0)
    avg_importance = df[top_k_features].mean(axis=0)
    var_importance = df[top_k_features].var(axis=0)

    plt.figure(figsize=(20, 12))
    plt.bar(
        avg_importance.index,
        avg_importance,
        yerr=var_importance,
        capsize=5,
        color="blue",
        alpha=0.7,
    )
    plt.xlabel("Features")
    plt.ylabel("Average Importance")
    plt.title(
        f"Average Feature Importance with Variance for {target_variable} (Random Forest)"
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_feature_time_series_rf(models, target_variable: str, top_k_features):
    """
    Plots the time series of feature importance for the top k features.

    Args:
        models (dict): Dictionary of Random Forest models by year.
        target_variable (str): Name of the target variable.
        top_k_features (list): List of top k features to plot.
    """
    feature_importances = {}
    r2_scores = {}

    # Format feature names for better readability
    def format_feature_name(feature):
        if "." in feature:
            metric, category = feature.split(".")
            # Get readable metric and category names from mappings if available
            if metric in metric_mapping:
                metric = metric_mapping[metric]
            if category in category_mapping:
                category = category_mapping[category]
            return f"{metric} - {category}"
        return feature

    # Create a mapping from original to formatted names
    formatted_names = {
        feature: format_feature_name(feature) for feature in top_k_features
    }
    formatted_target = format_feature_name(target_variable)

    for year, model_dict in models.items():
        model = model_dict["model"]
        feature_names = model_dict["X"].columns
        importances = model.feature_importances_
        feature_importances[year] = dict(zip(feature_names, importances))

        # Store R² scores from cross-validation results
        if "cv_results" in model_dict and "R²" in model_dict["cv_results"]:
            r2_scores[year] = model_dict["cv_results"]["R²"]["mean"]
        # Fallback to metrics if cv_results is not available
        elif "metrics" in model_dict and "R²" in model_dict["metrics"]:
            r2_scores[year] = model_dict["metrics"]["R²"]

    df = pd.DataFrame(feature_importances).T.fillna(0)
    df_top = df[top_k_features]

    # Define different markers
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "8"]

    # Create figure with increased width to accommodate both legend and R² text box
    fig, ax = plt.subplots(figsize=(22, 10))  # Increased width from 20 to 22

    # Plot each feature
    for i, feature in enumerate(top_k_features):
        marker = markers[i % len(markers)]
        ax.plot(
            df_top.index,
            df_top[feature],
            marker=marker,
            label=formatted_names[feature],
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Feature Importance", fontsize=14)
    if formatted_target == "Graduation Rate - Total":
        formatted_target = "Graduation Rate"
    elif formatted_target == "College Readiness Rate - Total":
        formatted_target = "College Readiness Rate"
    ax.set_title(
        f"Feature Importance Over Time for {formatted_target} (Random Forest)",
        fontsize=16,
    )

    # Create legend and place it on the right side
    ax.legend(
        title="Features",
        bbox_to_anchor=(1.02, 1),  # Moved slightly closer to the plot
        loc="upper left",
        fontsize=12,
        title_fontsize=14,
        edgecolor="black",
        borderaxespad=0.0,
    )

    # Add R² values text box with better positioning
    if r2_scores:
        sorted_years = sorted(r2_scores.keys())
        r2_text = "R² Values:\n" + "\n".join(
            [f"{year}: {r2_scores[year]:.4f}" for year in sorted_years]
        )
        avg_r2 = sum(r2_scores.values()) / len(r2_scores)
        r2_text += f"\n\nAverage: {avg_r2:.4f}"
        ax.text(
            0.92,
            0.72,
            r2_text,
            transform=ax.transAxes,
            fontsize=12,
            horizontalalignment="center",
            bbox=dict(boxstyle="round,pad=0.25", edgecolor="black", facecolor="white"),
        )

    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Adjust the layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Increased right margin from 0.75 to 0.85
    plt.show()


def cross_validate_random_forest(
    df,
    target_column,
    columns_to_drop,
    n_splits=5,
    random_state=42,
    print_results=True,
    print_plot=False,
    rf_params=None,
    n_jobs=1,
):
    """
    Performs k-fold cross-validation on a Random Forest regression model.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        columns_to_drop (list): List of column names to drop from the features.
        n_splits (int): Number of folds for cross-validation.
        random_state (int): Random seed for reproducibility.
        print_results (bool): Whether to print cross-validation results.
        rf_params (dict): Optional parameters for Random Forest model. If None, default parameters are used.

    Returns:
        dict: A dictionary containing cross-validation scores and the final trained model.
    """
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import (
        mean_squared_error,
        r2_score,
        mean_absolute_error,
        make_scorer,
    )
    import numpy as np

    # Prepare data
    X = df.drop(columns=columns_to_drop)
    X = X.select_dtypes(include=[np.number])  # Select only numeric columns
    y = df[target_column]

    # Align X and y by dropping rows where y is NaN
    y = pd.to_numeric(df[target_column], errors="coerce")
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Define default Random Forest parameters if not provided
    if rf_params is None:
        rf_params = {
            "n_estimators": 100,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }

    rf_regressor = RandomForestRegressor(**rf_params)

    rmse_scorer = make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=False,
    )
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    r2_scorer = make_scorer(r2_score, greater_is_better=True)

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
        "R²": {"mean": r2_scores.mean(), "std": r2_scores.std(), "scores": r2_scores},
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
        print(f"R²: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")

    if print_plot:
        plt.figure(figsize=(12, 6))

        metrics = ["RMSE", "MAE", "R²"]
        scores = [rmse_scores, mae_scores, r2_scores]
        colors = ["#FF9999", "#66B2FF", "#99CC99"]

        for i, (metric, score, color) in enumerate(zip(metrics, scores, colors)):
            plt.subplot(1, 3, i + 1)
            plt.boxplot(score, patch_artist=True, boxprops=dict(facecolor=color))
            plt.title(f"{metric} Cross-Validation for {target_column}")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

        plt.show()

    rf_regressor.fit(X, y)

    return {
        "model": rf_regressor,
        "cv_results": cv_results,
        "feature_names": X.columns.tolist(),
        "target_column": target_column,
        "X": X,
    }


def plot_cv_feature_importance_rf(cv_results, top_n=5):
    """
    Plots feature importances from cross-validated Random Forest models across multiple years.

    Args:
        cv_results (dict): A dictionary where keys are years and values are result dictionaries
                           from cross_validate_random_forest function.
        top_n (int): Number of top features to display.

    Returns:
        list: List of top feature names based on average importance across all years.
    """
    feature_importances = {}

    for year, cv_result in cv_results.items():
        model = cv_result["model"]
        feature_names = cv_result["feature_names"]
        importances = model.feature_importances_
        feature_importances[year] = dict(zip(feature_names, importances))

    df = pd.DataFrame(feature_importances).T.fillna(0)

    mean_importance = df.mean(axis=0).sort_values(ascending=False)
    top_features = mean_importance.head(top_n).index.tolist()
    top_features_importance = mean_importance.head(top_n)

    plt.figure(figsize=(12, 8))
    plt.title(f"Top {top_n} Feature Importances Across Years (Cross-Validated)")
    plt.barh(
        range(len(top_features)),
        top_features_importance[::-1],
        align="center",
    )
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel("Average Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    return top_features
