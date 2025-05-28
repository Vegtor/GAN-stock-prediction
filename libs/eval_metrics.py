import numpy as np
from scipy.stats import pearsonr

def trend_correlation(y_pred, y_true, window):
    y_true_ma = y_true.rolling(window).mean()
    y_pred_ma = y_pred.rolling(window).mean()
    mask = ~np.isnan(y_true_ma) & ~np.isnan(y_pred_ma)
    return pearsonr(y_true_ma[mask], y_pred_ma[mask])[0]  # Only correlation coefficient


def directional_accuracy(y_pred, y_true, window):
    y_true_ma = y_true.rolling(window).mean().diff().dropna()
    y_pred_ma = y_pred.rolling(window).mean().diff().dropna()
    correct = (np.sign(y_true_ma) == np.sign(y_pred_ma)).astype(int)
    return correct.mean()


def trend_mae(y_pred, y_true, window):
    y_true_ma = y_true.rolling(window).mean()
    y_pred_ma = y_pred.rolling(window).mean()
    return np.abs(y_true_ma - y_pred_ma).dropna().mean()


def relative_variability(y_pred, y_true, window):
    y_true_ma = y_true.rolling(window).mean().dropna()
    y_pred_ma = y_pred.rolling(window).mean().dropna()
    return y_pred_ma.std() / y_true_ma.std()
