import logging
from datetime import date, datetime, timedelta
from typing import Literal, NoReturn

import numpy as np
import pandas as pd
from dash import html
from gaw_qc.data.classes import ProcessedData, UserInput
from gaw_qc.data.modelling import (
    debias,
    downscaling,
    get_models,
    run_sublof,
)
from gaw_qc.data.preparation import (
    calculate_diurnal,
    calculate_thresholds,
    calculate_variability,
    filter_data,
    monthly_means,
    select_years_for_plots,
)
from gaw_qc.data.utils import list_to_class
from gaw_qc.db.variables import GawVars
from gaw_qc.log_utils.decorators import log_function
from gaw_qc.models.model_config import get_cams_vars, ModelSettings
from gaw_qc.time_utils.manipulation import limit_to_max_length, limit_to_one_year
from gaw_qc.time_utils.timestamps import parse_timestamps
from sqlalchemy import Engine

logger = logging.getLogger(__name__)


@log_function(logger)
def read_meta(engine: Engine) -> pd.DataFrame:
    """return the metadata of the stations that have data and add a column
    giving the variables available at each station
    :param engine: sqlalchemy engine
    :return: Data frame of metadata
    """
    variables = []
    with engine.connect() as conn:
        df_stations = pd.read_sql_query("SELECT * FROM stations", conn)
        df_series = pd.read_sql_query("SELECT * FROM series", conn)
    for cod in df_stations.gaw_id:
        stat_vars = ", ".join(
            df_series.loc[df_series.gaw_id==cod, "variable"].unique()
        ).upper()
        variables.append(stat_vars)
    df_stations["variables"] = variables

    return df_stations[df_stations.variables != ""]


@log_function(logger)
def read_series(engine: Engine, gaw_id: str) -> pd.DataFrame:
    """return the available variables, heights and contributors for a given station
    :param engine: sqlalchemy engine
    :param gaw_id: GAW id of the target station
    :return: Data frame with columns 'variable', 'height', 'contributor'
    """
    query = """
    SELECT variable, height, contributor
    FROM series
    WHERE gaw_id = :gaw_id
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Reading series for {gaw_id}")
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn, params=dict(gaw_id=gaw_id))


@log_function(logger)
def read_height(engine: Engine, gaw_id: str, v: GawVars, h: int) -> float:
    """return the sampling height
    :param engine: sqlalchemy engine
    :param gaw_id: GAW id of the target station
    :param v: Variable
    :param h: Height index
    :return: Height in metres
    """
    query = f"""
        SELECT height FROM series
        WHERE gaw_id = :gaw_id AND variable = :variable
        ORDER BY height, contributor
        LIMIT 1 OFFSET {h}
        """
    with engine.connect() as conn:
        return pd.read_sql_query(
            query, conn, params=dict(gaw_id=gaw_id, variable=v)
        ).squeeze()


@log_function(logger)
def read_contributor(
        engine: Engine, gaw_id: str, v: GawVars, h: int
) -> list[str, html.B]:
    """return the contributor, if not available return an empty string
    :param engine: sqlalchemy engine
    :param gaw_id: GAW id of the target station
    :param v: Variable
    :param h: Height index
    :return: Data contributor
    """
    query = f"""
        SELECT contributor, export FROM series
        WHERE gaw_id = :gaw_id AND variable = :variable
        ORDER BY height, contributor
        LIMIT 1 OFFSET {h}
        """
    with engine.connect() as conn:
        contr = pd.read_sql_query(
            query, conn, params=dict(gaw_id=gaw_id, variable=v)
        )
        if contr.contributor.item() == "":
            label = ""
        else:
            label = ["Data contributor: ", html.B(contr.contributor.item())]
            if contr.export.item() == 0:
                label = label + [" (export not available)"]

        return label


@log_function(logger)
def read_policy(engine: Engine, gaw_id: str, v: GawVars, h: int) -> int:
    """return the data export policy
    :param engine: sqlalchemy engine
    :param gaw_id: GAW id of the target station
    :param v: Variable
    :param h: Height index
    :return: 0 for no export allowed, 1 for export allowed
    """
    query = f"""
        SELECT export FROM series
        WHERE gaw_id = :gaw_id AND variable = :variable
        ORDER BY height, contributor
        LIMIT 1 OFFSET {h}
        """
    with engine.connect() as conn:
        return pd.read_sql_query(
            query, conn, params=dict(gaw_id=gaw_id, variable=v)
        ).squeeze()


@log_function(logger)
def read_start_end(
    engine: Engine, gaw_id: str, v: GawVars, h: int
) -> tuple[date, date]:
    """return the first and last date of a series
    :param engine: sqlalchemy engine
    :param gaw_id: GAW id of the target station
    :param v: Variable
    :param h: Height index
    :return: List of two dates
    """
    query = f"""
    WITH ids AS (
        SELECT id FROM series
        WHERE gaw_id = :gaw_id AND variable = :variable
        ORDER BY height, contributor
        LIMIT 1 OFFSET {h}
    )

    SELECT
        min(first_value) AS first_time,
        max(last_value) AS last_time
    FROM
    (
        SELECT
            MIN(time) AS first_value, MAX(time) AS last_value
        FROM gaw_hourly
        WHERE  series_id IN ids
        UNION ALL
        SELECT
            MIN(time) AS first_value, MAX(time) AS last_value
        FROM gaw_monthly
        WHERE  series_id IN ids
    )
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Reading timespan for {gaw_id}, {v}, {h}")
    with engine.connect() as conn:
        vals = pd.read_sql_query(
            query, conn, params=dict(gaw_id=gaw_id, variable=v)
        )
        #logger.info(f"Found timespan: {vals}")
        start, end = parse_timestamps(vals)
        return start.date(), end.date()


@log_function(logger)
def get_cams_summary(engine: Engine, gaw_id: str) -> pd.DataFrame:
    """Returns the stats of the CAMS series for a given station
    :param engine: sqlalchemy engine
    :param gaw_id: GAW id of the target station
    :return: Data frame with columns 'last_entry' and 'first_entry'
    """
    query = """
    with filter as(
        select
            variable, height, gaw_id, id
        from series
        where gaw_id = :series_id
    )

    select min(time) as first_entry, max(time) as last_entry, variable, height from gaw_hourly
    join filter on filter.id = gaw_hourly.series_id
    group by series_id
    union all
    select min(time) as first_entry, max(time) as last_entry, variable, height from gaw_monthly
    join filter on filter.id = gaw_monthly.series_id
    group by series_id
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Reading CAMS stats for {gaw_id}")
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn, params=dict(gaw_id=gaw_id))


@log_function(logger)
def read_end_date_cams(engine:Engine) -> str:
    """
    Return the last date for CAMS data
    (read only the first series to speed up)
    """
    query = """
        SELECT MAX(time) FROM cams_hourly WHERE series_id = (
            SELECT series_id FROM cams_hourly LIMIT 1
        )
        """
    with engine.connect() as conn:
        last_val = pd.read_sql(query, conn)
        return datetime.strftime(parse_timestamps(last_val)[0], "%d %b %Y")


@log_function(logger)
def read_end_date_gaw(engine:Engine) -> str:
    """
    Return the last date for GAW data
    """
    query = "SELECT MAX(time) FROM gaw_hourly"
    with engine.connect() as conn:
        last_val = pd.read_sql(query, conn)
        return datetime.strftime(parse_timestamps(last_val)[0], "%d %b %Y")


@log_function(logger)
def get_cams_data(
    engine: Engine, gaw_id: str, v: GawVars, h: int, monthly: bool, cams_vars: str
) -> pd.DataFrame:
    """
    Read CAMS data from the database
    """
    if monthly:
        tb = "cams_monthly"
        variables = "time, value"
    else:
        tb = "cams_hourly"
        variables = "time, value, " + cams_vars
    query = f"""
    WITH series_id AS (
        SELECT id FROM series
        WHERE gaw_id = :gaw_id AND variable = :variable
        ORDER BY height, contributor
        LIMIT 1 OFFSET {h}
    )
    SELECT
        {variables}
    FROM {tb}
    WHERE series_id in series_id
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Reading CAMS data for {gaw_id}, {v}, {h}")
    with engine.connect() as conn:
        return pd.read_sql_query(
            query, conn, params=dict(gaw_id=gaw_id, variable=v)
        )


@log_function(logger)
def get_gaw_data(
    engine: Engine, gaw_id: str, v: GawVars, h: int, monthly: bool
) -> pd.DataFrame:
    """
    Read GAW data from the database
    In case of duplicates, `h` determines which series to extract
    """
    tb = "gaw_monthly" if monthly else "gaw_hourly"
    query = f"""
    WITH series_id AS (
        SELECT id FROM series
        WHERE gaw_id = :gaw_id AND variable = :variable
        ORDER BY height, contributor
        LIMIT 1 OFFSET {h}
    )
    SELECT
        time, value, n_meas
    FROM {tb}
    WHERE series_id in series_id
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Reading GAW data for {gaw_id}, {v}, {h}")
    with engine.connect() as conn:
        return pd.read_sql_query(
            query, conn, params=dict(gaw_id=gaw_id, variable=v)
        )


@log_function(logger)
def read_data(
    engine: Engine, gaw_id: str, v: GawVars, h: int, cams_on: bool, cams_vars: str
) -> tuple[pd.DataFrame, Literal["hourly", "monthly"]]:
    """Read data for the target station from the database
    :param engine: sqlalchemy engine
    :param gaw_id: GAW id of the target station
    :param v: Variable
    :param h: Height index
    :param cams_on: whether to read CAMS data
    :param cams_vars: columns to read from CAMS data
    :return: Data frame of hourly or monthly data, and a string giving the time resolution
    """
    # Try hourly first
    gaw_data = get_gaw_data(engine, gaw_id, v, h, False)
    res = "hourly"
    if len(gaw_data) == 0:
        gaw_data = get_gaw_data(engine, gaw_id, v, h, True)
        res = "monthly"

    out_gaw = gaw_data.rename(columns={"value": v})
    if cams_on:
        cams_data = get_cams_data(
            engine, gaw_id, v, h, res == "monthly", cams_vars
        )
        if len(cams_data) == 0:
            logger.warning("CAMS data not found for this station")
            out = out_gaw
        else:
            # Increase time resolution through linear interpolation
            cams_data["value"].interpolate(
                inplace=True
            )
            cams_data.rename(
                columns={"value": v + "_cams", "value_tc": "tc_" + v},
                inplace=True
            )
            out = pd.merge(out_gaw, cams_data, how="outer", on="time")
    else:
        out = out_gaw

    out.sort_values(["time"], inplace=True)
    out["time"] = pd.to_datetime(out["time"], unit="s")
    out["hour"] = out["time"].dt.hour
    out["doy"] = out["time"].dt.dayofyear
    out["trend"] = np.arange(out.shape[0])
    out.set_index("time", inplace=True)

    return out, res


@log_function(logger)
def get_data(
    engine: Engine, uinput: UserInput
) -> ProcessedData | list[NoReturn] | None:
    """
    Read data from database and perform the most expensive computations
    Return an empty list if the uploaded file is not readable
    Return None if no data are found in the database for the selected period
    """
    logger.info("Getting data")
    par = uinput.par.lower()

    # Calculate offset from UTC
    if (uinput.tz == "UTC") | (uinput.content is None):
        tdiff = 0.0
    else:
        tdiff = float(uinput.tz[3:6])

    # Parse uploaded data (time is converted to UTC)
    if uinput.content is None:
        time_start_utc = datetime.strptime(uinput.date_start, "%Y-%m-%d")
        time_end_utc = datetime.strptime(uinput.date_end, "%Y-%m-%d")
        is_new = False
    else:
        df_up = uinput.parse_data()
        if not isinstance(df_up, pd.DataFrame):
            return []  # file could not be read
        time_start_utc = df_up.index[0]
        time_end_utc = df_up.index[-1]
        is_new = True

    # Clamp dates
    time_start_utc = limit_to_one_year(time_start_utc, time_end_utc)
    if uinput.content is not None:
        df_up = df_up[df_up.index >= time_start_utc]

    # Define local start and end times
    time_start = time_start_utc + timedelta(hours=tdiff)
    time_end = time_end_utc + timedelta(hours=tdiff)

    # Load models
    model_config = ModelSettings()
    subLOF, ml_model = get_models(model_config)

    # Are CAMS data needed?
    use_cams = (
        (get_cams_vars(par) != "")
        & (time_end_utc >= model_config.min_date_cams)
    )

    # Read data from db and remove values based on insufficient measurements
    df_all, res = read_data(
        engine,
        uinput.cod,
        par,
        uinput.hei,
        use_cams,
        get_cams_vars(par),
    )
    df_all = filter_data(df_all, res, 0.25, 15, 5)
    logger.info(f"Data available with {res} resolution")

    # Set use_cams to False if CAMS data are missing
    use_cams = (use_cams) & (par + "_cams" in df_all.columns)

    # Calculate monthly means of uploaded data when resolution in the db is monthly
    if (res == "monthly") & (uinput.content is not None):
        df_up = monthly_means(
            df_up, model_config.n_min, time_start_utc, time_end_utc
        )

    # Merge uploaded data with historical data
    if uinput.content is not None:
        df_all.loc[
            (
                (df_all.index >= time_start_utc)
                & (df_all.index <= time_end_utc)
            ),
            par
        ] = np.nan
        if use_cams:
            df_all.loc[df_all.index.isin(df_up.index), par] = df_up.loc[
                df_up.index.isin(df_all.index), par
            ]
        # Concatenate data and drop duplicated times (keep the last instance)
        # This guarantees that no data are lost in the merging
        df_all = pd.concat([df_all, df_up])
        df_all = df_all.groupby(df_all.index).last().sort_index()

    # Calculate monthly means of merged data
    if res == "hourly":
        df_mon = monthly_means(
            df_all[[par]],
            model_config.n_min,
            time_start,
            time_end,
        )
        if use_cams:
            df_mon[par + "_cams"] = monthly_means(
                df_all[[par + "_cams"]],
                model_config.n_min,
                time_start,
                time_end,
            )
    else:
        df_mon = df_all[[par]].copy()
        if use_cams:
            df_mon[par + "_cams"] = df_all[par + "_cams"]

    # Define test data set
    if uinput.content is None:
        df_test = df_all[
            (
                (df_all.index >= time_start)
                & (df_all.index < time_end + timedelta(days=1))
            )
        ].drop(columns="n_meas") # add one day to include last day selected
        local_times = df_test.index
    else:
        df_test = df_all[
            (df_all.index >= time_start_utc) & (df_all.index <= time_end_utc)
        ].drop(columns="n_meas")
        local_times = df_test.index.shift(tdiff, freq="H")

    # Check if there is any data available in the test period
    if df_test[par].count() == 0:
        logger.info("No data available in the target period")
        return None

    # Define training data set
    t_start, t_end = limit_to_max_length(
        df_all[par].dropna().index,
        model_config.max_months_ml * 30 * 24,
        df_test.index[0],
        df_test.index[-1]
    )
    df_train = df_all[(df_all.index >= t_start) & (df_all.index <= t_end)].copy()
    df_train.loc[df_train.index.isin(df_test.index), par] = np.nan
    df_train.drop(columns="n_meas", inplace=True)
    logger.info(f"Data between {t_start} and {t_end} will be used for training")

    # Downscale/debias CAMS
    y_pred_mon, cams_score, cams_score_train = 3 * [pd.Series()]
    if use_cams:
        if res == "hourly":
            y_pred, y_pred_mon, cams_score = downscaling(
                df_train,
                df_test,
                par,
                model_config.window_size_cams,
                model_config.n_min,
                ml_model,
            )
            y_pred_mon = y_pred_mon[y_pred_mon.index.isin(df_mon.index)]
            y_pred_mon = y_pred_mon.round(2)
            cams_score_train = cams_score[
                cams_score.index.isin(df_train.dropna().index)
            ]
            if len(y_pred) > 0:
                df_test["CAMS+"] = y_pred.round(2).values
            else:
                df_test["CAMS+"] = np.nan
        else:
            y_pred_mon = debias(df_train, df_test[par + "_cams"], par).round(2)

    # Apply Sub-LOF on training and full period
    if res == "hourly":
        lof_score_train = run_sublof(df_train[par], subLOF, model_config.window_size)
        lof_score_all = run_sublof(df_all[par], subLOF, model_config.window_size)
        lof_score_test = lof_score_all[lof_score_all.index.isin(df_test.index)]
    else:
        lof_score_train = lof_score_test = pd.Series([])

    # Calculate flagging thresholds for each strictness level
    df_thresholds = calculate_thresholds(
        cams_score_train,
        lof_score_train,
        df_train[par].max(),
        df_train[par].min(),
        res,
    )

    # Prepare data for monthly plot
    if res == "hourly":
        n_meas = df_test[par].groupby(pd.Grouper(freq="1M", label="left")).count()
        n_meas.index = n_meas.index + timedelta(days=1)
    else:
        n_meas = df_all["n_meas"]
    df_monplot = df_mon[
        df_mon.index <= local_times[-1]
    ].copy()  # exclude data after target period for monthly plot
    n_meas = n_meas[n_meas.index <= local_times[-1]]
    df_monplot["n"] = np.nan
    df_monplot.loc[df_monplot.index.isin(n_meas.index), "n"] = n_meas

    # Select years for cycle plots
    years = select_years_for_plots(df_all[par])
    first_year = years[-1] - model_config.n_years_max
    years = years[years >= first_year]

    # Rearrange monthly data into a seasonal cycle data frame
    columns = list(years.drop(local_times.year[-1]).astype(str)) + ["test"]
    df_sc = pd.DataFrame(columns=columns, index=np.arange(1, 13))
    for c in columns:
        if c == "test":
            df_mon_c = df_mon[
                (df_mon.index >= local_times[0]) & (df_mon.index <= local_times[-1])
            ]
        else:
            df_mon_c = df_mon[df_mon.index.year == int(c)]
        df_sc.loc[df_mon_c.index.month, c] = df_mon_c[par].values

    # Calculate diurnal cycle and variability cycle
    df_dc = calculate_diurnal(
        df_all.loc[df_all.index.year >= first_year, par],
        local_times,
        tdiff,
        years,
        res,
    )
    df_vc = calculate_variability(
        df_all.loc[df_all.index.year >= first_year, par],
        df_sc,
        local_times,
        years,
        res,
    )

    # Wrap everything together and convert to json format for storage
    test_cols = [par]
    if use_cams:
        test_cols.append(par + "_cams")
        if res == "hourly":
            test_cols.append("CAMS+")
    out = [
        uinput.cod,
        par,
        uinput.hei,
        res,
        is_new,
        time_start_utc,
        time_end_utc,
        tdiff,
        years[-1],
        df_test[test_cols].to_json(date_format="iso", orient="columns"),
        df_mon.to_json(date_format="iso", orient="columns"),
        df_monplot.to_json(date_format="iso", orient="columns"),
        y_pred_mon.to_json(date_format="iso", orient="index"),
        cams_score.to_json(date_format="iso", orient="index"),
        lof_score_test.to_json(date_format="iso", orient="index"),
        df_thresholds.to_json(orient="columns"),
        df_dc.to_json(orient="columns"),
        df_sc.to_json(orient="columns"),
        df_vc.to_json(orient="columns"),
    ]

    return list_to_class(out, ProcessedData)
