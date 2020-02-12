###############################################################################
import sklearn.metrics as mets
###############################################################################
METRIC_NAMES = ["max_error", "explained_variance_score", "mean_absolute_error", "mean_squared_error", "mean_squared_log_error", "median_absolute_error", "r2_score", "mean_poisson_deviance", "mean_gamma_deviance", "mean_tweedie_deviance"]
def set_metrics(y_pred, y_true, dict):
    try:
        dict["max_error"] = mets.max_error(y_true, y_pred)
    except:
        pass
    try:
        dict["explained_variance_score"] = mets.explained_variance_score(y_true, y_pred)
    except:
        pass
    try:
        dict["mean_absolute_error"] = mets.mean_absolute_error(y_true, y_pred)
    except:
        pass
    try:
        dict["mean_squared_error"] = mets.mean_squared_error(y_true, y_pred)
    except:
        pass
    try:
        dict["mean_squared_log_error"] = mets.mean_squared_log_error(y_true, y_pred)
    except:
        pass
    try:
        dict["median_absolute_error"] = mets.median_absolute_error(y_true, y_pred)
    except:
        pass
    try:
        dict["r2_score"] = mets.r2_score(y_true, y_pred)
    except:
        pass
    try:
        dict["mean_poisson_deviance"] = mets.mean_poisson_deviance(y_true, y_pred)
    except:
        pass
    try:
        dict["mean_gamma_deviance"] = mets.mean_gamma_deviance(y_true, y_pred)
    except:
        pass
    try:
        dict["mean_tweedie_deviance"] =  mets.mean_tweedie_deviance(y_true, y_pred)
    except:
        pass
    return dict
###############################################################################
