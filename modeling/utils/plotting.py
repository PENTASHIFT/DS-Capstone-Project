import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence


plt.style.use("default")

### Mapping dictionaries for readability
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


### XGBoost Plotting Functions


def plot_residuals_xgb(
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


def plot_predicted_vs_actual_xgb(
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


def plot_longitudinal_top_features_xgb(models, model_type, target_variable: str, k=5):
    feature_importances = {}

    for year, model in models.items():
        importance = model["model"].get_booster().get_score(importance_type="gain")
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

    plt.title(f"Top {k} Features for {target_variable} ({model_type})")
    plt.ylabel("Feature Importance")
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


def plot_top_features_xgb(cv_results, target_variable, top_n=10, figsize=(12, 8)):
    """
    Plots the top features by importance across all years in a horizontal bar chart.

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

    avg_importances = {feature: values for feature, values in importances.items()}
    top_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)[
        :top_n
    ]

    top_features = top_features[::-1]

    feature_names = [format_feature_name(f[0]) for f in top_features]
    importance_values = [f[1] for f in top_features]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = range(len(feature_names))
    ax.barh(
        y_pos,
        importance_values,
        align="center",
        color="navy",
        edgecolor="navy",
        alpha=0.8,
    )

    model_stats = (
        f"RMSE: {cv_results['metrics']['RMSE']['mean']:.4f} ± {cv_results['metrics']['RMSE']['std']:.4f}\n"
        f"MAE: {cv_results['metrics']['MAE']['mean']:.4f} ± {cv_results['metrics']['MAE']['std']:.4f}\n"
        f"R²: {cv_results['metrics']['R^2']['mean']:.4f} ± {cv_results['metrics']['R^2']['std']:.4f}"
    )

    ax.text(
        0.98,
        0.05,
        model_stats,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top Features by Importance for {target_variable} (XGBoost)")
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

    return [
        f[0]
        for f in sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
    ]


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


def plot_longitudinal_feature_importance_xgb(cv_results, top_n=5):
    """
    Plots feature importances from a dictionary of XGBoost models.

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


def plot_feature_time_series_xgb(
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


### Random Forest Plotting Functions


def plot_residuals_rf(y_test, y_pred, target_column):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color="blue")
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"Random Forest Residual Plot: {target_column}")
    plt.show()


def plot_actual_predicted_rf(y_test, y_pred, target_column):
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


def plot_longtiudinal_top_features_rf(models, target_variable: str, k=5):
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
        bbox_to_anchor=(1.02, 1), 
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

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


def plot_longitudinal_feature_importance_rf(cv_results, top_n=5):
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

    r2_key = "R^2" if "R^2" in cv_results["metrics"] else "R²"

    model_stats = (
        f"RMSE: {cv_results['metrics']['RMSE']['mean']:.4f} ± {cv_results['metrics']['RMSE']['std']:.4f}\n"
        f"MAE: {cv_results['metrics']['MAE']['mean']:.4f} ± {cv_results['metrics']['MAE']['std']:.4f}\n"
        f"R²: {cv_results['metrics'][r2_key]['mean']:.4f} ± {cv_results['metrics'][r2_key]['std']:.4f}"
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


def plot_feature_interaction(model, X, feature1_idx, feature2_idx, feature_names=None):
    """
    Creates a 2D interaction plot between two features.
    
    Args:
        model: Trained XGBoost or Random Forest model
        X: Feature dataset
        feature1_idx: Index of first feature
        feature2_idx: Index of second feature
        feature_names: Optional list of feature names
    """
    if feature_names is None:
        feature_names = X.columns.tolist()
    
    ### Create meshgrid
    x1 = np.linspace(X.iloc[:, feature1_idx].min(), X.iloc[:, feature1_idx].max(), 50)
    x2 = np.linspace(X.iloc[:, feature2_idx].min(), X.iloc[:, feature2_idx].max(), 50)
    X1, X2 = np.meshgrid(x1, x2)
    
    ### Create prediction dataset
    X_pred = X.iloc[0:1].copy()
    X_pred = pd.concat([X_pred] * (50 * 50), ignore_index=True)
    
    idx = 0
    for i in range(50):
        for j in range(50):
            X_pred.iloc[idx, feature1_idx] = X1[i, j]
            X_pred.iloc[idx, feature2_idx] = X2[i, j]
            idx += 1
    
    ### Make predictions
    y_pred = model.predict(X_pred)
    

    Z = y_pred.reshape(50, 50)
    
    plt.figure(figsize=(12, 8))
    plt.contourf(X1, X2, Z, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Predicted Value')
    plt.xlabel(feature_names[feature1_idx])
    plt.ylabel(feature_names[feature2_idx])
    plt.title(f'Interaction Between {format_feature_name(feature_names[feature1_idx])} and {format_feature_name(feature_names[feature2_idx])}')
    plt.tight_layout()
    plt.show()
