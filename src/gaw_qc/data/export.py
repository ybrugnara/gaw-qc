import logging
import numpy as np
import pandas as pd
import pytz
from datetime import datetime
from sqlalchemy import Engine
from gaw_qc.data.classes import ProcessedData
from gaw_qc.data.preparation import (
    get_n_months_to_predict, process_cycles, process_monthly
)
from gaw_qc.data.sources import read_contributor, read_height
from gaw_qc.db.variables import GawUnits, GawSources
from gaw_qc.log_utils.decorators import log_function

logger = logging.getLogger(__name__)


@log_function(logger)
def add_header(
        engine: Engine,
        df: pd.DataFrame,
        cod: str,
        par: str,
        hei: int,
        is_new: bool,
        cams_on: bool
    ) -> str:
    """ Add header with data credits to export
    :param df: Data frame to export
    :param cod: Station ID
    :param par: Variable
    :param hei: Sampling height index
    :param is_new: Whether the analyzed data were uploaded by the user
    :param cams_on: Whether CAMS data are exported
    :return: String of comma-separated data ready for export
    """
    height = read_height(engine, cod, par, hei)
    contr = read_contributor(engine, cod, par, hei)
    header = (
        "# Mole fraction of "
        + par.upper()
        + " in " + GawUnits[par].value
        + " at " + cod
        + " (" + str(height) + " m)."
    )
    if not is_new:
        header += " GAW data source: " + GawSources[par].value
        if contr != "":
            header += " - Contributor: " + contr[1].children
        header += "."
    if cams_on & ("CAMS" in df.columns):
        header += " CAMS data source: Copernicus Atmosphere Data Store (ADS - atmosphere.copernicus.eu/data)."
    header += (
        " File created by the GAW-QC app on "
        + datetime.now(pytz.timezone("UTC")).isoformat(sep=" ", timespec="seconds")
        + "."
    )
    out = header + "\r\n" + df.to_csv(index=False)

    return out


@log_function(logger)
def export_hourly(
        engine: Engine,
        data: ProcessedData,
        test_data_flagged: str,
        cams_on: bool,
        selected_q: int
) -> tuple[str, datetime, datetime]:
    # Read data from json
    df_test_flagged = pd.read_json(test_data_flagged, orient="columns")

    # Create data frame for export
    df_exp = pd.DataFrame(
        {
            "Time": df_test_flagged.index,
            "Value": df_test_flagged[data.par],
            "CAMS": np.nan,
            "CAMS+": np.nan,
            "Flag LOF": df_test_flagged["Flag LOF"],
            "Flag CAMS": np.nan,
            "Strictness": selected_q,
        }
    )

    # Fill CAMS columns or drop them if CAMS is not used
    if cams_on:
        for c, c_exp in zip(
                [data.par + "_cams", "CAMS+", "Flag CAMS"],
                ["CAMS", "CAMS+", "Flag CAMS"]
                ):
            if c in df_test_flagged.columns:
                df_exp[c_exp] = df_test_flagged[c]
            else:
                df_exp.drop(columns=c_exp, inplace=True)
    else:
        df_exp.drop(columns=["CAMS", "CAMS+", "Flag CAMS"], inplace=True)

    out = add_header(
        engine,
        df_exp,
        data.cod,
        data.par,
        data.hei,
        data.is_new,
        cams_on,
    )

    logger.info(f"Exporting hourly data ({data.cod}, {data.par}, {data.hei})")

    return out, df_test_flagged.index[0], df_test_flagged.index[-1]


@log_function(logger)
def export_monthly(
        engine: Engine,
        data: ProcessedData,
        cams_on: bool,
        trend: dict[str, str],
        max_length: int
) -> tuple[str, datetime, datetime]:
    # Read data from json
    df_mon = pd.read_json(data.monthly_data, orient="columns")
    df_monplot = pd.read_json(data.monthly_data_plot, orient="columns")
    y_pred_mon = pd.read_json(data.cams_plus_mon, orient="index", typ="series")
    mtp = get_n_months_to_predict(df_monplot.index, data.time_start)

    # Get SARIMA prediction and flags
    df_flagged = process_monthly(
        df_monplot, mtp, max_length, trend["value"], data.par
    )

    # Create data frame for export
    length = np.min([df_mon.shape[0], max_length * 12 + mtp])
    df_exp = pd.DataFrame(
        {
            "Time": df_flagged.index[-mtp:],
            "Value": df_flagged[data.par].iloc[-mtp:],
            "N measurements": df_flagged["n"].iloc[-mtp:],
            "SARIMA (best estimate)": df_flagged["prediction"].iloc[-mtp:],
            "SARIMA (lower limit)": df_flagged["lower"].iloc[-mtp:],
            "SARIMA (upper limit)": df_flagged["upper"].iloc[-mtp:],
            "CAMS": np.nan,
            "CAMS+": np.nan,
            "Flag": 0,
            "Years used": int((length - mtp) / 12),
            "Trend": trend["label"],
        }
    )
    df_exp.loc[
        df_exp["Time"].isin(df_flagged.index[df_flagged["flag"]]), "Flag"
    ] = 1

    # Fill CAMS columns or drop them if CAMS is not used
    if cams_on:
        if (data.par + "_cams") in df_flagged.columns:
            df_exp["CAMS"] = df_flagged[data.par + "_cams"].iloc[-mtp:]
        else:
            df_exp.drop(columns="CAMS", inplace=True)
        if len(y_pred_mon) > 0:
            df_exp["CAMS+"] = y_pred_mon
        else:
            df_exp.drop(columns="CAMS+", inplace=True)
    else:
        df_exp.drop(columns=["CAMS", "CAMS+"], inplace=True)

    # Add header and convert to csv
    out = add_header(
        engine,
        df_exp,
        data.cod,
        data.par,
        data.hei,
        data.is_new,
        cams_on,
    )

    logger.info(f"Exporting monthly data ({data.cod}, {data.par}, {data.hei})")

    return out, df_flagged.index[-mtp], df_flagged.index[-1]


@log_function(logger)
def export_diurnal(
        engine: Engine,
        data: ProcessedData,
        n_years: int
) -> str:
    export_data = process_cycles(data, n_years)

    # Rearrange data frame
    df_exp = export_data.diurnal_cycle.copy()
    df_exp["Hour"] = df_exp.index
    df_exp.rename(
        columns={
            "test": export_data.label,
            "test_n": "N days",
            "multiyear": export_data.period_label,
        },
        inplace=True,
    )
    df_exp = df_exp[
        ["Hour", export_data.label, "N days", export_data.period_label] +
        list(export_data.years_for_mean)
    ]

    # Remove multi-year average if there is only one year
    if len(export_data.years_for_mean) == 1:
        df_exp.drop(columns=[export_data.period_label], inplace=True)
        
    # Add header and convert to csv
    out = add_header(engine, df_exp, data.cod, data.par, data.hei, False, False)

    logger.info(f"Exporting diurnal cycle ({data.cod}, {data.par}, {data.hei})")

    return out


@log_function(logger)
def export_seasonal(
        engine: Engine,
        data: ProcessedData,
        n_years: int
) -> str:
    export_data = process_cycles(data, n_years)

    # Rearrange data frame
    df_exp = export_data.seasonal_cycle.copy()
    df_exp["Month"] = df_exp.index
    df_exp.rename(
        columns={
            "test": export_data.label,
            "multiyear": export_data.period_label,
        },
        inplace=True,
    )
    df_exp = df_exp[
        ["Month", export_data.label, export_data.period_label] +
        list(export_data.years_for_mean)
    ]

    # Remove multi-year average if there is only one year
    if len(export_data.years_for_mean) == 1:
        df_exp.drop(columns=[export_data.period_label], inplace=True)

    # Add header and convert to csv
    out = add_header(engine, df_exp, data.cod, data.par, data.hei, False, False)

    logger.info(f"Exporting seasonal cycle ({data.cod}, {data.par}, {data.hei})")

    return out


@log_function(logger)
def export_variability(
        engine: Engine,
        data: ProcessedData,
        n_years: int
) -> str:
    export_data = process_cycles(data, n_years)

    # Rearrange data frame
    df_exp = export_data.var_cycle.copy()
    df_exp["Month"] = df_exp.index
    df_exp.rename(
        columns={
            "test": export_data.label,
            "multiyear": export_data.period_label,
        },
        inplace=True,
    )
    df_exp = df_exp[
        ["Month", export_data.label, export_data.period_label] +
        list(export_data.years_for_mean)
    ]

    # Remove multi-year average if there is only one year
    if len(export_data.years_for_mean) == 1:
        df_exp.drop(columns=[export_data.period_label], inplace=True)

    # Add header and convert to csv
    out = add_header(engine, df_exp, data.cod, data.par, data.hei, False, False)
    out = out.replace("# Mole", "# Standard deviation of mole")

    logger.info(f"Exporting variability cycle ({data.cod}, {data.par}, {data.hei})")

    return out


@log_function(logger)
def build_filename(engine: Engine, data: ProcessedData, suffix: str) -> str:
    height = read_height(engine, data.cod, data.par, data.hei)
    if data.is_new:
        contr = "upload"
    else:
        contr = read_contributor(engine, data.cod, data.par, data.hei)
        if contr != "":
            contr = contr[1].children

    return "_".join(
        [
            data.cod,
            data.par,
            str(height),
            contr,
            suffix
        ]
    )
