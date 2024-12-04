import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from scipy import stats

def rfe_feature_selection(data, target, n_features_to_select=10, ranking_threshold=None, train_size=0.7,
                          return_details=False):
    """Perform Recursive Feature Elimination (RFE) for feature selection.

    Args:
        data (pandas.DataFrame): DataFrame containing features.
        target (list): List of target variables.
        n_features_to_select (int, optional): Number of features to select.
            Defaults to 10.
        ranking_threshold (float, optional): Threshold for feature ranking.
            Defaults to None.
        train_size (float, optional): Proportion of data for training.
            Defaults to 0.7.
        return_details (bool, optional): Whether to return additional details
            about selection. Defaults to False.

    Returns:
        List or tuple containing selected features or tuple with selected features
        and detailed selection information.
    """
    X = data.copy()
    y = target.copy()
    split_index = int(len(X) * train_size)
    X_train, X_test = X.iloc[:split_index, :], X.iloc[split_index:, :]
    y_train, y_test = y[:split_index], y[split_index:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_model = LinearRegression()
    rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select)
    rfe.fit(X_train_scaled, y_train)

    X_train_rfe = rfe.transform(X_train_scaled)
    X_test_rfe = rfe.transform(X_test_scaled)

    model = LinearRegression()
    model.fit(X_train_rfe, y_train)

    y_pred = model.predict(X_test_rfe)
    mse = mean_squared_error(y_test, y_pred)

    feature_selection_details = {'selected_features_default': np.array(X.columns)[rfe.support_],
                                 'feature_ranking': rfe.ranking_, 'model_mse': mse, 'model_coefficients': model.coef_}

    if ranking_threshold is not None:
        selected_features = np.array(X.columns)[rfe.ranking_ <= ranking_threshold]
    else:
        selected_features = np.array(X.columns)[rfe.support_]

    if return_details:
        return selected_features, feature_selection_details
    return selected_features


# Example usage
# selected_features = rfe_feature_selection(data, target)
# selected_features, details = rfe_feature_selection(data, target, return_details=True)

def advanced_anova_feature_selection(data, target, alpha=0.05, normalize=True, return_details=False):
    """
    Advanced ANOVA feature selection with additional preprocessing options

    Args:
        data (pandas.DataFrame): DataFrame containing features.
        target (list): List of target variables.
        alpha (float, optional): Significance level for feature selection.
            Defaults to 0.05.
        normalize (bool, optional): Whether to normalize the data to have zero mean.
            Defaults to True.
        return_details (bool, optional): Whether to return additional details about selection.
            Defaults to False.

    Returns:
        List or tuple containing selected features or tuple with selected features
        and detailed selection information.
    """

    X = data.copy()
    y = target.copy()

    if normalize:
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    feature_analysis = {}

    for feature in X.columns:
        if pd.api.types.is_numeric_dtype(X[feature]):
            bins = pd.qcut(X[feature], q=4, labels=False, duplicates='drop')
            groups = [y[bins == category] for category in range(len(bins.unique()))]
        else:
            groups = [y[X[feature] == category] for category in X[feature].unique()]

        try:
            f_statistic, p_value = stats.f_oneway(*groups)
            feature_analysis[feature] = {'p_value': p_value, 'f_statistic': f_statistic, 'significant': p_value < alpha}
        except Exception as e:
            feature_analysis[feature] = {'p_value': 1.0, 'f_statistic': 0, 'significant': False, 'error': str(e)}

    selected_features = [feature for feature, analysis in feature_analysis.items() if analysis['significant']]
    if return_details:
        return selected_features, feature_analysis
    return selected_features

# Example usage
# selected_features = advanced_anova_feature_selection(data, 'target_column')
# selected_features, details = advanced_anova_feature_selection(data, 'target_column', return_details=True)

"""
OLD VERSION **************************************************************************************
def rfe_selection(data, target, ranking_threshold):
    X = data.copy()
    y = target.copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()

    n_features_to_select = 10
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)

    rfe.fit(X_train, y_train)

    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)

    model.fit(X_train_rfe, y_train)

    y_pred = model.predict(X_test_rfe)
    mse = mean_squared_error(y_test, y_pred)

    #print(f"Selected Features: {np.array(X.columns)[rfe.support_]}")
    #print(f"Model Mean Squared Error: {mse:.4f}")
    #print(f"Feature Ranking: {rfe.ranking_}")
    #print(f"Model Coefficients: {model.coef_}")
    ranking = rfe.ranking_

    #ranking_threshold = 60
    features_better_than_threshold = np.array(X.columns)[ranking <= ranking_threshold]
    #print(f"Features with ranking better than {ranking_threshold}: {features_better_than_threshold}")
    return features_better_than_threshold
********************************************************************************************************
"""