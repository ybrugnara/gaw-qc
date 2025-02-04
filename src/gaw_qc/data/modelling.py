import logging
from datetime import timedelta

from gaw_qc.models.model_config import ModelSettings
from gaw_qc.models.models import regression_model
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from pyod.models.lof import LOF
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from gaw_qc.log_utils.decorators import log_function
logger = logging.getLogger(__name__)


@log_function(logger)
def aggregate_scores(scores: pd.Series, w_size: int) -> pd.Series:
    """assign score to each point as the average of the scores of the windows it is in
    :param scores: Series produced by run_sublof
    :param w_size: Window size (integer)
    :return: Series of aggregated scores
    """
    added_times = pd.date_range(
        scores.index[-1] + timedelta(hours=1),
        scores.index[-1] + timedelta(hours=w_size - 1),
        freq="H",
    )
    scores = pd.concat([scores, pd.Series(np.nan, index=added_times)])

    return scores.rolling(w_size, min_periods=1).mean()


@log_function(logger)
def run_sublof(series: pd.Series, model: LOF, win: int) -> pd.Series:
    """run Sub-LOF algorithm
    :param series: Data series to analyze (must be complete - no times missing)
    :param model: Instance of the LOF model to use
    :param win: Window size (integer)
    :return: Series of anomaly scores
    """
    input_data = pd.DataFrame(
        sliding_window_view(series.values, window_shape=win)
    )
    is_nan = input_data.isna().any(axis=1)
    input_data = input_data[~is_nan]
    times = series[: -win + 1].index[~is_nan]
    model.fit(input_data)
    scores = pd.Series(model.decision_scores_, index=times)
    scores = scores.reindex(series.index)
    scores = aggregate_scores(scores, win)

    return scores


@log_function(logger)
def forecast_sm(
    df_m: pd.DataFrame,
    n: int,
    max_l: int,
    sel_t: str,
    p_conf: float,
    min_months: int
) -> pd.DataFrame:
    """make prediction using a SARIMA model
    :param df_m: Data frame with monthly data (one variable and time as index)
    :param n: Number of time steps to predict
    :param max_l: Maximum length in years of the training period
    :param sel_t: Type of trend
    :param p_conf: Confidence level (probability)
    :param min_months: Months of data required to fit the model
    :return: Data frame with measurements, prediction, and confidence interval
    """
    # Extract periods and parameter name
    length = np.min([df_m.shape[0], max_l * 12 + n])
    par = df_m.columns[0]
    df_sar = df_m.iloc[-length:,:]

    # Define output data frame
    df_prediction = pd.DataFrame(
        columns=["prediction", "upper", "lower"],
        index=df_sar.index[-n:],
    )

    if df_sar[par].iloc[:-n].count() >= min_months:
        # Define and fit SARIMA model
        sarfit = sm.tsa.statespace.SARIMAX(
            df_sar[par].iloc[:-n],
            order=ModelSettings().sarima_order,
            seasonal_order=ModelSettings().sarima_sorder,
            trend=sel_t,
            enforce_stationarity=ModelSettings().sarima_stationarity,
        ).fit(disp=False)

        # Make forecast
        prediction = sarfit.get_forecast(n)

        # Populate data frame
        df_prediction["prediction"] = prediction.predicted_mean.values.round(2)
        df_confidence = prediction.conf_int(alpha=p_conf)
        df_prediction["upper"] = df_confidence["upper " + par].values.round(2)
        df_prediction["lower"] = df_confidence["lower " + par].values.round(2)

    return pd.concat([df_sar, df_prediction], axis=1)


@log_function(logger)
def debias(df_train: pd.DataFrame, cams: pd.Series, par: str) -> pd.Series:
    """debias monthly CAMS data using linear regression (one model for each calendar month)
    :param df_train: Data frame of training data (containing both measurements and CAMS data, with time as index)
    :param cams: Series of CAMS data to debias (time as index)
    :param par: Variable to debias (defines the column names)
    :return: Series of debiased CAMS data
    """
    LR = LinearRegression()
    df_diff = df_train[par] - df_train[par + "_cams"]
    df_diff.dropna(inplace=True)
    if len(df_diff) < ModelSettings().min_months_ml:
        return pd.Series()

    # Fit a LR model for each month
    months = np.unique(cams.index.month)
    cams_debiased = cams.copy()
    for m in months:
        df_m = df_diff[df_diff.index.month == m]
        X = df_m.index.year.to_numpy().reshape(-1, 1)
        LR.fit(X, df_m)
        X = cams[cams.index.month == m].index.year.to_numpy().reshape(-1, 1)
        cams_debiased[cams_debiased.index.month == m] = (
            cams[cams.index.month == m] + LR.predict(X)
        )

    return cams_debiased


@log_function(logger)
def downscaling(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    par: str,
    w: int,
    n_min: int,
    model: RegressorMixin,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """dowscaling algorithm for CAMS forecasts; the anomaly score is calculated as the moving median of the prediction error
    :param df_train: Data frame of training data (containing both measurements and CAMS data, with time as index)
    :param df_test: Same as df_train but for the target period
    :param par: Variable to debias (defines the column names)
    :param w: Size of the moving window used to calculate the anomaly score (integer)
    :param n_min: Number of valid hourly values required to calculate monthly means
    :param model: Instance of a sklearn regression model
    :return: Series of downscaled data for the target period (hourly and monthly), series of the anomaly score
    """
    df_train_cams = df_train.dropna()
    df_test_cams = df_test.drop(par, axis=1).dropna()
    if (
            (df_train_cams.shape[0] < ModelSettings().min_months_ml * 30 * 24)
            | (df_test_cams.shape[0] == 0)
    ):
        return pd.Series(), pd.Series(), pd.Series()

    # Fit model
    model.fit(
        df_train_cams.drop(par, axis=1).to_numpy(),
        df_train_cams[par].to_numpy()
    )
    y_pred = model.predict(df_test_cams.to_numpy())
    y_pred = pd.Series(
        y_pred, index=df_test_cams.index
    ).reindex(index=df_test.index)

    # Monthly data (for SARIMA plot)
    y_pred_mon = y_pred.groupby(pd.Grouper(freq="1M", label="left")).mean()
    y_pred_n = y_pred.groupby(pd.Grouper(freq="1M", label="left")).count()
    y_pred_mon[y_pred_n < n_min] = np.nan
    y_pred_mon.index = y_pred_mon.index + timedelta(days=1)

    # Anomaly score
    x_to_predict = pd.concat(
        [df_train_cams.drop(par, axis=1), df_test_cams]
    ).sort_index()
    y_pred_all = model.predict(x_to_predict.to_numpy())
    y_pred_all = pd.Series(y_pred_all, index=x_to_predict.index)
    y_to_compare = pd.concat(
        [df_train_cams[par],
         df_test.loc[df_test.index.isin(df_test_cams.index), par]]
    ).sort_index()
    errors = y_pred_all - y_to_compare
    all_times = pd.concat([df_train, df_test]).sort_index().index
    errors = errors.reindex(index=all_times[~all_times.duplicated()])
    errors_train = errors[errors.index.isin(df_train_cams.index)]
    diff_series = errors - np.median(errors_train)
    anom_score = diff_series.rolling(w, min_periods=int(w / 2)).median()

    return y_pred, y_pred_mon, anom_score


@log_function(logger)
def get_models(config: ModelSettings) -> tuple[LOF, RegressorMixin]:
    """
    Load models for the Sub-LOF and the downscaling algorithm
    given a configuration object
    """

    # Define LOF instance
    subLOF = LOF(n_neighbors=config.n_neighbors, metric="euclidean")

    # Define regression model
    ml_model = regression_model()

    return subLOF, ml_model
