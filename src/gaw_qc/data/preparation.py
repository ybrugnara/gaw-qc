import logging
from datetime import datetime, timedelta
import calendar
from typing import Literal

import numpy as np
import pandas as pd
from gaw_qc.data.classes import ProcessedData, PlottingData
from gaw_qc.data.modelling import forecast_sm
from gaw_qc.data.utils import list_to_class
from gaw_qc.db.variables import GawVars, GawUnits
from gaw_qc.models.model_config import get_cams_vars, ModelSettings

from gaw_qc.log_utils.decorators import log_function
logger = logging.getLogger(__name__)


@log_function(logger)
def filter_data(
    df: pd.DataFrame,
    res: Literal["hourly", "monthly"],
    thr_h: float,
    thr_m: float,
    w: int,
) -> pd.DataFrame:
    """assign NaN to hourly/monthly means based on an insufficient number of measurements
    :param df: Data frame produced by read_data
    :param res: temporal resolution ('hourly' or 'monthly')
    :param thr_h: Minimum fraction of measurements required in an hour with respect of a moving maximum
    :param thr_m: Minimum number of days required in a month
    :param w: Window size for moving maximum in hourly data
    :return: Data frame with NaNs in place of data based on insufficient measurements (column n_meas is dropped)
    """
    par = df.columns[0]
    if res == "hourly":
        running_max = df["n_meas"].rolling(w, min_periods=1, center=True).max()
        df.loc[(df["n_meas"] > 0) & (df["n_meas"] < thr_h * running_max), par] = np.nan
    else:
        df.loc[(df["n_meas"] > 0) & (df["n_meas"] < thr_m), par] = np.nan

    return df


@log_function(logger)
def add_missing(timeseries: pd.DataFrame) -> pd.DataFrame:
    """fill missing times with NaN for hourly data
    :param timeseries: Data frame with hourly resolution (time as index)
    :return: Filled in data frame
    """
    return timeseries.resample('H').asfreq()


@log_function(logger)
def monthly_means(
        timeseries: pd.DataFrame, 
        n_min: int, 
        t_start: datetime,
        t_end: datetime
) -> pd.DataFrame:
    """calculate monthly means from hourly data (require at least n_min values per month)
    Data at the beginning or end of a month that is partly outside the target period
    are excluded, i.e. only complete months are considered
    :param timeseries: Data frame with one variable and time as index
    :param n_min: Minimum number of valid hourly values
    :param t_start: Start time of target period
    :param t_end: End time of target period
    :return: Data frame of monthly means
    """
    t_firstmonth = datetime(t_start.year, t_start.month, 1, 0)
    t_lastmonth = datetime(
        t_end.year,
        t_end.month,
        calendar.monthrange(t_end.year, t_end.month)[1],
        23
    )
    to_drop = (
        ((timeseries.index >= t_firstmonth) & (timeseries.index < t_start)) |
        ((timeseries.index <= t_lastmonth) & (timeseries.index > t_end))
    )
    tmp = timeseries.drop(index=timeseries.index[to_drop])
    out = tmp.groupby(pd.Grouper(freq="1M", label="left")).mean().round(2)
    n = tmp.groupby(pd.Grouper(freq="1M", label="left")).count()
    out[n < n_min] = np.nan
    out.index = out.index + timedelta(days=1)  # first day of the month as index

    return out


@log_function(logger)
def process_hourly(
    proc_data: ProcessedData,
    cams_on: bool,
    selected_q: int,
) -> pd.DataFrame:
    """
    Flag hourly data for plot and export
    """
    # Read from json
    df_test = pd.read_json(proc_data.test_data, orient="columns")
    score_cams = pd.read_json(proc_data.anom_score_cams, orient="index", typ="series")
    score_lof = pd.read_json(proc_data.anom_score_lof, orient="index")
    thresholds = pd.read_json(proc_data.thresholds, orient="columns")
    param = proc_data.par

    # Flag data after Sub-LOF
    df_test["Flag LOF"] = 0
    thr_yellow = thresholds.loc[selected_q, "LOF"]
    thr_red = thr_yellow * 2
    flags_yellow = df_test[
        (score_lof[0] > thr_yellow) & (score_lof[0] <= thr_red)
    ]
    flags_red = df_test[score_lof[0] > thr_red]
    df_test.loc[df_test.index.isin(flags_yellow.index), "Flag LOF"] = 1

    # Flag extreme outliers that exceed historical records by half the historical range (red flags only)
    flags_red = pd.concat(
        [
            flags_red,
            df_test[
                (df_test[param] > thresholds.loc[1, "range upper"])
                | (df_test[param] < thresholds.loc[1, "range lower"])
            ],
        ]
    )
    flags_red = flags_red.groupby(flags_red.index).first()  # drop duplicates
    df_test.loc[df_test.index.isin(flags_red.index), "Flag LOF"] = 2

    # Flag data after CAMS (yellow only; two yellow flags [LOF+CAMS] trigger a red flag)
    if (
        cams_on &
        (get_cams_vars(param) != "") &
        (proc_data.time_start >= ModelSettings().min_date_cams) &
        (len(score_cams) > 0)
    ):
        df_test["Flag CAMS"] = 0
        anom_score_test = score_cams[score_cams.index.isin(df_test.index)]
        flags_cum_yellow = (
            anom_score_test < thresholds.loc[selected_q, "CAMS lower"]
        ) | (anom_score_test > thresholds.loc[selected_q, "CAMS upper"])
        flags_cum_yellow = pd.Series(df_test.index[flags_cum_yellow])
        # Extend flags to the entire window used by cumulative_score
        for fl in flags_cum_yellow:
            t0 = max(
                fl - timedelta(hours=ModelSettings().window_size_cams - 1),
                df_test.index[0],
            )
            additional_times = pd.Series(pd.date_range(t0, fl, freq="H"))
            flags_cum_yellow = pd.concat(
                [flags_cum_yellow, additional_times], ignore_index=True
            )
        flags_cum_yellow.drop_duplicates(inplace=True)
        flags_cum_yellow.sort_values(inplace=True)
        df_test.loc[df_test.index.isin(flags_cum_yellow), "Flag CAMS"] = 1
        df_test.loc[df_test[param].isna(), "Flag CAMS"] = 0
        df_test["Flag all"] = df_test["Flag LOF"] + df_test["Flag CAMS"]
    else:
        df_test["Flag all"] = df_test["Flag LOF"]

    # Convert to json for caching
    out = df_test.to_json(date_format="iso", orient="columns")

    return out


@log_function(logger)
def process_monthly(
    df_monplot: pd.DataFrame,
    mtp: int,
    max_length: int,
    selected_trend: str,
    param: GawVars,
) -> pd.DataFrame:
    """
    Fit SARIMA model and flag monthly data for plot and export
    """
    # Make forecast with SARIMA
    df_out = forecast_sm(
        df_monplot,
        mtp,
        max_length,
        selected_trend,
        ModelSettings().p_conf,
        ModelSettings().min_months_sarima,
    )

    # Add flags
    flags = (df_out[param] > df_out["upper"]) | (df_out[param] < df_out["lower"])
    df_out["flag"] = flags

    return df_out


@log_function(logger)
def process_cycles(data: ProcessedData, n_years: int) -> PlottingData:
    """
    Add multi-year averages and create labels for cycles plots and exports
    """
    # Read data from json
    df_dc = pd.read_json(data.diurnal_cycle, orient="columns")
    df_sc = pd.read_json(data.seasonal_cycle, orient="columns")
    df_vc = pd.read_json(data.var_cycle, orient="columns")

    # Choose years
    first_year = data.last_year - n_years
    years = df_sc.columns.drop("test")
    years_for_mean = years[years >= str(first_year)]
    years_to_drop = years[years < str(first_year)]

    # Work on diurnal cycle
    if data.res == "hourly":
        df_dc.drop(columns=years_to_drop, inplace=True)
        df_dc["multiyear"] = (
            df_dc[years_for_mean].mean(axis=1).round(2).values
        )

    # Work on seasonal cycle
    df_sc.drop(columns=years_to_drop, inplace=True)
    df_sc["multiyear"] = (
        df_sc[years_for_mean].mean(axis=1).round(2).values
    )

    # Work on variability cycle
    if data.res == "hourly":
        df_vc.drop(columns=years_to_drop, inplace=True)
        df_vc["multiyear"] = (
            df_vc[years_for_mean].mean(axis=1).round(2).values
        )

    # Create labels
    label = "Your data" if data.is_new else "Selected period"
    period_label = (
        "Mean " +
        min(years_for_mean) +
        "-" +
        max(years_for_mean)
    )
    y_title = (
        data.par.upper() +
        " mole fraction (" +
        GawUnits[data.par].value +
        ")"
    )

    out = [
        label,
        period_label,
        y_title,
        years_for_mean,
        df_dc,
        df_sc,
        df_vc,
    ]

    return list_to_class(out, PlottingData)


@log_function(logger)
def get_n_months_to_predict(
        index: pd.DatetimeIndex,
        start: datetime
) -> int:
    """
    Figure out how many months must be predicted by SARIMA
    """
    return len(index[index >= start.replace(hour=0)])


@log_function(logger)
def select_years_for_plots(
        s: pd.Series, res: Literal["hourly", "monthly"]
) -> pd.Index:
    """
    Figure out which years are relevant for the plots
    """
    s_clean = s.dropna()
    years = s_clean.index.year.unique()
    if res == "hourly":
        # Drop edge years that have less than one day of data
        if len(s_clean[s_clean.index.year==years[0]]) < 24:
            years = years[1:]
        if len(s_clean[s_clean.index.year==years[-1]]) < 24:
            years = years[:-1]

    return years


@log_function(logger)
def calculate_thresholds(
    cams_score_train: pd.Series,
    lof_score_train: pd.Series,
    data_max: float,
    data_min: float,
    res: Literal["hourly", "monthly"],
) -> pd.DataFrame:
    """
    Calculate anomaly score thresholds for flags for each strictness level
    """
    model_config = ModelSettings()
    levels = np.arange(1, model_config.n_levels+1)

    # Define output data frame
    df_thresholds = pd.DataFrame(
        columns=["LOF", "CAMS lower", "CAMS upper", "range lower", "range upper"]
    )

    # Calculate threshold for Sub-LOF
    if res == "hourly":
        p_thrs_lof = model_config.thr0_lof + model_config.incr_lof * levels
        df_thresholds["LOF"] = np.quantile(lof_score_train.dropna(), p_thrs_lof)
    else:
        df_thresholds["LOF"] = [np.nan, np.nan, np.nan]

    # Calculate thresholds for CAMS
    if (len(cams_score_train) > 0) & (res == "hourly"):
        p_thrs_cams = model_config.thr0_cams + model_config.incr_cams * levels
        df_thresholds["CAMS lower"] = (
            np.quantile(cams_score_train.dropna(), 1 - p_thrs_cams) * 2
        )
        df_thresholds["CAMS upper"] = (
            np.quantile(cams_score_train.dropna(), p_thrs_cams) * 2
        )

    # Calculate thresholds for large outliers
    if res == "hourly":
        half_range = (data_max - data_min) / 2
        df_thresholds["range lower"] = data_min - half_range
        df_thresholds["range upper"] = data_max + half_range

    return df_thresholds.set_index(levels)


@log_function(logger)
def calculate_seasonal(
    data: pd.Series,
    times: pd.Index, # time index of test period in the original time zone
    years: pd.Index,
) -> pd.DataFrame:
    """
    Rearrange monthly data into a seasonal cycle data frame
    """
    columns = list(years.drop(times.year[-1]).astype(str)) + ["test"]
    df_sc = pd.DataFrame(columns=columns, index=np.arange(1, 13))
    for c in columns:
        if c == "test":
            data_c = data[
                (data.index >= times[0].replace(hour=0))
                & (data.index <= times[-1])
            ]
        else:
            data_c = data[data.index.year == int(c)]
        df_sc.loc[data_c.index.month, c] = data_c.values

    return df_sc


@log_function(logger)
def calculate_diurnal(
    data: pd.Series,
    times: pd.Index, # time index of test period in the original time zone
    tdiff: float,
    years: pd.Index,
    res: Literal["hourly", "monthly"],
) -> pd.DataFrame:
    """
    Calculate mean diurnal cycle for each year
    """
    # Define output data frame
    columns = list(years.drop(times.year[-1]).astype(str)) + ["test"]
    df_dc = pd.DataFrame(columns=columns, index=np.arange(24))

    if res == "hourly":

        # Convert to local time and extract relevant julian days
        if tdiff != 0:
            data.index = data.index.shift(tdiff, freq="H")
        j_days = times.dayofyear.unique()
        data_sub = data[data.index.dayofyear.isin(j_days)]

        # Shift data before 01.01 by one year
        if times.year[0] < times.year[-1]:
            last_j = times.dayofyear[-1]
            new_index = pd.Series(data_sub.index)
            new_index[data_sub.index.dayofyear > last_j] += pd.DateOffset(years=1)
            data_sub.index = new_index

        # Calculate diurnal cycle for each year
        for iy in years:
            data_year = data_sub[data_sub.index.year == iy]
            c = "test" if iy == times.year[-1] else str(iy)
            df_dc[c] = data_year.groupby(data_year.index.hour).mean().round(2)
            df_dc[c + "_n"] = data_year.groupby(data_year.index.hour).count()

    return df_dc


@log_function(logger)
def calculate_variability(
    data: pd.Series,
    df_sc: pd.DataFrame,
    times: pd.Index, # time index of test period in the original time zone
    years: pd.Index,
    res: Literal["hourly", "monthly"],
) -> pd.DataFrame:
    """
    Calculate variability cycle for each year
    Missing values are synchronized with the seasonal cycle data frame
    """
    # Define output data frame
    columns = list(years.drop(times.year[-1]).astype(str)) + ["test"]
    df_vc = pd.DataFrame(columns=columns, index=np.arange(1,13))

    if res == "hourly":

        # Fill in data frame
        for c in columns:
            if c == "test":
                data_c = data[
                    (data.index >= times[0].replace(hour=0))
                    & (data.index <= times[-1])
                ]
            else:
                data_c = data[data.index.year == int(c)]
            vc = data_c.groupby(data_c.index.month).std().round(2)
            months_to_nan = df_sc.index[df_sc[c].isna()]
            vc[vc.index.isin(months_to_nan)] = np.nan
            df_vc.loc[vc.index, c] = vc.values

    return df_vc
