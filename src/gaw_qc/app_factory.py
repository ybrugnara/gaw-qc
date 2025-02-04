import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tempfile
import warnings
from argparse import ArgumentParser
from datetime import date, datetime
from pathlib import Path
from dash import Dash, Input, Output, State, callback
from dash.exceptions import PreventUpdate
from flask_caching import Cache
from plotly.subplots import make_subplots
from sqlalchemy import Engine, create_engine

from gaw_qc.config.app_config import AppConfig
from gaw_qc.data.classes import ProcessedData, UserInput
from gaw_qc.data.export import (
    build_filename,
    export_diurnal,
    export_hourly,
    export_monthly,
    export_seasonal,
    export_variability,
)
from gaw_qc.data.preparation import (
    get_n_months_to_predict,
    process_cycles,
    process_hourly,
    process_monthly
)
from gaw_qc.data.sources import (
    get_data,
    read_contributor,
    read_height,
    read_end_date_cams,
    read_end_date_gaw,
    read_meta,
    read_policy,
    read_series,
    read_start_end
)
from gaw_qc.data.utils import list_to_class
from gaw_qc.db.variables import GawVars, GawUnits
from gaw_qc.log_utils.decorators import log_function
from gaw_qc.models.model_config import get_cams_vars, ModelSettings
from gaw_qc.pages.main import layout
from gaw_qc.plotting.aesthetics import add_logo, PlotSettings
from gaw_qc.plotting.base_plots import empty_plot
from gaw_qc.plotting.map import plot_map_of_stations
from gaw_qc.plotting.plots import (
    plot_diurnal,
    plot_histogram,
    plot_hourly,
    plot_monthly,
    plot_pie,
    plot_seasonal,
    plot_variability,
)

warnings.filterwarnings("ignore")


# Initialize logger
logger = logging.getLogger(__name__)


# Define Dash app
def create_app(
    engine: Engine,
    cache: Cache,
    assets_path: Path,
    theme: str,
    title: str,
) -> Dash:
    """
    Factory function that creates a Dash App given a `Cache` and a SQlAlchemy `Engine`.
    The advantage is that the module is now side-effect free.
    We pass `engine` and `cache` to the app to allow dependency injection, so that later
    we can replace the filesystem cache with a Redis cache, or the SQLite database with a PostgreSQL database.
    """
    app = Dash(
        external_stylesheets=[theme],
        assets_folder=assets_path.as_posix(),
        title=title,
        update_title="Loading...",
    )

    # Load model settings
    model_config = ModelSettings()
    logger.debug(f"Model settings: {model_config}")

    # Set the application to the cache
    cache.init_app(app.server)
    logger.info(f"Creating app with cache: {cache.config}")

    # Define some functions
    @app.server.route("/test", methods=["GET"])
    @log_function(logger)
    def test_app() -> str:
        """
        A test route to check if the app is running.
        This is useful to check if the container is built correctly.
        """
        logger.info("Test route")
        meta = read_meta(engine)
        return "Test route OK"

    @cache.memoize()
    def get_cached_meta(engine: Engine) -> pd.DataFrame:
        """
        Cached version of read_meta to avoid reading the same metadata from
        the database over again
        """
        return read_meta(engine)

    @cache.memoize()
    def get_cached_series(engine: Engine, stat: str) -> pd.DataFrame:
        """
        Cached version of read_series to avoid reading the same metadata from
        the database over again
        """
        return read_series(engine, stat)

    @cache.memoize()
    def get_cached_data(
            engine: Engine,
            input_list: list[str|float|None]
    ) -> ProcessedData | None:
        """
        Perform the most expensive computations and cache the results
        """
        user_input = list_to_class(input_list, UserInput)
        return get_data(engine, user_input)

    @cache.memoize()
    def reprocess_hourly(
        proc_data: ProcessedData,
        cams_on: bool,
        selected_q: int,
    ) -> pd.DataFrame:
        """
        Cached version of process_hourly
        """
        return process_hourly(proc_data, cams_on, selected_q)

    # Read list of stations
    metadata = get_cached_meta(engine)
    stations = (
        metadata[["name", "gaw_id"]]
        .set_index("gaw_id")
        .sort_values(by="name")
        .to_dict()["name"]
    )

    # Set layout of the app by calling the layout factory
    logger.info("Loading layout")
    app.layout = layout.main_layout(
        stations, read_end_date_cams(engine), read_end_date_gaw(engine)
    )


    # Define callbacks
    @app.callback(
        Output("modal-credits", "is_open"),
        [Input("open-credits", "n_clicks")],
        [State("modal-credits", "is_open")],
    )
    @log_function(logger)
    def toggle_credits(n: int, is_open: bool) -> bool:
        if n:
            return not is_open
        return is_open


    @app.callback(
        Output("modal-acknowl", "is_open"),
        [Input("open-acknowl", "n_clicks")],
        [State("modal-acknowl", "is_open")],
    )
    @log_function(logger)
    def toggle_acknowledgments(n: int, is_open: bool) -> bool:
        if n:
            return not is_open
        return is_open


    @app.callback(
        Output("modal-report-bug", "is_open"),
        [Input("open-report-bug", "n_clicks")],
        [State("modal-report-bug", "is_open")],
    )
    @log_function(logger)
    def toggle_report_bug(n: int, is_open: bool) -> bool:
        if n:
            return not is_open
        return is_open


    @callback(
        Output("param-dropdown", "options", allow_duplicate=True),
        Output("param-dropdown", "value", allow_duplicate=True),
        Output("height-dropdown", "value", allow_duplicate=True),
        Output("stat-par-hei", "data", allow_duplicate=True),
        Input("station-dropdown", "value"),
        prevent_initial_call=True,
    )
    @log_function(logger)
    def update_variables(
        stat: str | None,
    ) -> tuple[list[GawVars], None, None, list[str | None]]:
        meta = get_cached_series(engine, stat)
        pars = map(str.upper, np.sort(np.unique(meta["variable"])))
        store = [stat, None, None]

        return list(pars), None, None, store


    @callback(
        Output("height-dropdown", "options", allow_duplicate=True),
        Output("height-dropdown", "value", allow_duplicate=True),
        Output("stat-par-hei", "data", allow_duplicate=True),
        Input("param-dropdown", "value"),
        Input("height-dropdown", "value"),
        Input("stat-par-hei", "data"),
        prevent_initial_call=True,
    )
    @log_function(logger)
    def update_heights(
        par: GawVars,
        hei_index: int,
        stored_input: list[str | int | None],
    ) -> tuple[list[dict[str, int | float]], float | None, list[str | float | None]]:
        if par is None:
            raise PreventUpdate

        stat = stored_input[0]
        last_par = stored_input[1]
        meta = get_cached_series(engine, stat)
        sorted_meta = meta[meta["variable"] == par.lower()].sort_values(
            by=["height", "contributor"]
        )
        heights = sorted_meta["height"]

        h_options = [
            {'label': h, 'value': i} for i, h in zip(range(len(heights)), heights)
        ]
        new_hei = hei_index if par == last_par else None
        store = [stat, par, new_hei]

        return h_options, new_hei, store


    @callback(
        Output("date-range", "min_date_allowed"),
        Output("date-range", "max_date_allowed"),
        Output("date-range", "initial_visible_month"),
        Output("loading-dates", "display", allow_duplicate=True),
        Input("stat-par-hei", "data"),
        prevent_initial_call=True,
    )
    @cache.memoize()
    @log_function(logger)
    def update_dates(
        stored_input: list[str | int | None],
    ) -> tuple[str | None, str | None, str | None, str]:
        cod, par, hei_index = stored_input
        if (cod is None) | (par is None) | (hei_index is None):
            raise PreventUpdate

        start, end = read_start_end(engine, cod, par.lower(), hei_index)
        hei = read_height(engine, cod, par.lower(), hei_index)
        if pd.isnull(end):
            logger.error(f"No data found for {cod}, {par}, {hei}")
            return (None, None, None, "hide")

        logger.info(f"Data for {cod} - {par} - {hei} available from {start} to {end}")

        return (
            str(start),
            str(end),
            str(date(end.year, 1, 1)),
            "auto",
        )


    @callback(
        Output("label-contributor", "children"),
        Input("stat-par-hei", "data"),
        prevent_initial_call=True,
    )
    @log_function(logger)
    def update_contributor(stored_input: list[str | int | None]) -> str:
        cod, par, hei_index = stored_input
        if (cod is None) | (par is None) | (hei_index is None):
            raise PreventUpdate

        return read_contributor(engine, cod, par.lower(), hei_index)


    @callback(
        Output("timezone-dropdown", "value", allow_duplicate=True),
        Input("timezone-dropdown", "value"),
        Input("upload-data", "contents"),
        prevent_initial_call=True,
    )
    @log_function(logger)
    def update_tz(tz: str | None, content: str | None) -> str:
        if (tz is not None) | (content is None):
            raise PreventUpdate

        return "UTC"


    @callback(
        Output("map-of-stations", "figure"),
        Input("station-dropdown", "value"),
    )
    @log_function(logger)
    def update_map(cod: str | None) -> go.Figure:
        return plot_map_of_stations(get_cached_meta(engine), cod)


    @callback(
        Output("label-wait", "children"),
        Output("label-wait", "style"),
        Output("loading-dates", "display", allow_duplicate=True),
        Input("station-dropdown", "value"),
        Input("param-dropdown", "value"),
        Input("height-dropdown", "value"),
        Input("date-range", "end_date"),
        Input("timezone-dropdown", "value"),
        Input("upload-data", "contents"),
        prevent_initial_call=True,
    )
    @log_function(logger)
    def wait_message(
        cod: str | None,
        par: GawVars | None,
        hei_index: int | None,
        date1: str | None,
        tz: str | None,
        content: str | None,
    ) -> tuple[str, dict[str, str], str]:
        if (
            (cod is None)
            | (par is None)
            | (hei_index is None)
            | ((date1 is None) & ((tz is None) | (content is None)))
        ):
            return "", {"display": "none"}, "auto"
        else:
            return (
                "Please wait - Dashboard is loading...",
                {"font-size": "18px", "font-weight": "bold", "margin-bottom": "25px"},
                "hide",
            )


    # Reset dcc.Store to avoid firing all callbacks at once when an input parameter is changed
    @callback(
        Output("input-data", "data", allow_duplicate=True),
        Input("label-wait", "children"),
        prevent_initial_call=True,
    )
    @log_function(logger)
    def reset_data(lab: str) -> list[str]:
        if lab == "":
            raise PreventUpdate

        return []


    @callback(
        Output("input-data", "data", allow_duplicate=True),  # this fires the dashboard
        Output("date-range", "start_date", allow_duplicate=True),  # reset start date
        Output("date-range", "end_date", allow_duplicate=True),  # reset end date
        Output("upload-data", "contents", allow_duplicate=True),  # reset uploaded file
        Input("station-dropdown", "value"),
        Input("param-dropdown", "value"),
        Input("height-dropdown", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("timezone-dropdown", "value"),
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
        prevent_initial_call=True,
    )
    def update_data(
        cod: str | None,
        par: GawVars | None,
        hei_index: int | None,
        date_start: str | None,
        date_end: str | None,
        tz: str | None,
        content: str | None,
        filename: str | None,
    ) -> tuple[list[str | float | None] | None, None, None, None]:
        if (
            (cod is None)
            | (par is None)
            | (hei_index is None)
            | ((date_end is None) & ((tz is None) | (content is None)))
        ):
            raise PreventUpdate

        init_time = datetime.now()
        input_data = [
            cod, par, hei_index, date_start, date_end, tz, content, filename
        ]
        data = get_cached_data(engine, input_data)
        stop_time = datetime.now()

        if data is None:
            return None, None, None, None

        # Write log
        if data != []:
            to_log = [
                cod,
                par,
                hei_index,
                data.time_start,
                data.time_end,
                int(data.is_new),
                (stop_time - init_time).seconds,
            ]
            logger.info("data:" + ",".join(map(str, to_log)))

        return input_data, None, None, None


    @callback(
        Output("btn_csv_hourly", "disabled"),
        Output("btn_csv_monthly", "disabled"),
        Output("btn_csv_dc", "disabled"),
        Output("btn_csv_sc", "disabled"),
        Output("btn_csv_vc", "disabled"),
        Input("input-data", "data"),
    )
    def toggle_export_buttons(
            input_data: list[str] | None
    ) -> tuple[bool, bool, bool, bool, bool]:
        if (input_data == []) or (input_data is None):
            raise PreventUpdate

        cod, par, h = input_data[:3]
        content = input_data[6]
        policy = read_policy(engine, cod, par.lower(), h)

        return 5 * (not (bool(policy) | (content is not None)),)


    @callback(
        Output("graph-hourly", "figure"),
        Output("div-switch", "style"),
        Output("div-hourly", "style"),
        Output("div-hourly-settings", "style"),
        Output("div-hourly-info", "style"),
        Output("div-text-1", "style"),
        Input("points-switch", "on"),
        Input("cams-switch-1", "on"),
        Input("threshold-slider", "value"),
        Input("bin-slider", "value"),
        Input("input-data", "data"),
    )
    def update_figure_1(
        points_on: bool,
        cams_on: bool,
        selected_q: int,
        bin_size: float,
        input_data: list[str] | None,
    ) -> tuple[
        go.Figure,
        dict[str, str],
        dict[str, str],
        dict[str, str],
        dict[str, str],
        dict[str, str],
    ]:
        if input_data == []:
            raise PreventUpdate

        if input_data is None:
            return (
                empty_plot(
                    "No data available in the selected period - Try choosing a longer period"
                ),
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )

        # Get data from cache
        cache_data = get_cached_data(engine, input_data)

        if cache_data == []:
            return (
                empty_plot(
                    "Impossible to read uploaded file - Try to change the time format or check the wiki"
                ),
                {"display": "none"},
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "none"},
            )

        if cache_data.res == "monthly":
            return (
                empty_plot("No hourly data available"),
                {"display": "flex"},
                {"display": "block"},
                {"display": "none"},
                {"display": "none"},
                {"display": "flex"},
            )

        # Prepare data for hourly plot
        df_test_flagged = reprocess_hourly(cache_data, cams_on, selected_q)
        df_test_flagged = pd.read_json(df_test_flagged, orient="columns")

        # Define plot panels
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{"rowspan": 2}, {"type": "domain"}], [None, {}]],
            column_widths=[0.75, 0.25],
            row_heights=[0.5, 0.5],
            horizontal_spacing=0.075,
            vertical_spacing=0.075,
            subplot_titles=(" ", "", ""),
        )

        # Should CAMS data be plotted?
        use_cams = (
            cams_on &
            (get_cams_vars(cache_data.par) != "") &
            (cache_data.time_end >= ModelSettings().min_date_cams)
        )

        # Plot time series
        fig = plot_hourly(
            fig,
            df_test_flagged,
            cache_data.par,
            use_cams,
            points_on,
            row=1,
            col=1,
        )

        # Plot pie chart
        fig = plot_pie(fig, df_test_flagged, cache_data.par, row=1, col=2)

        # Plot histogram
        fig = plot_histogram(
            fig,
            df_test_flagged,
            cache_data.par,
            bin_size,
            use_cams,
            row=2,
            col=2,
        )

        # Add logo Empa
        fig = add_logo(fig, 0.0, 0.0, 0.075, 0.2)

        # Add title, legend, margins
        meta = metadata[metadata["gaw_id"] == cache_data.cod]
        fig.update_layout(
            title={
                "text": meta["name"].item() + " (" + cache_data.cod + "): hourly means",
                "xanchor": "center",
                "x": 0.5,
            },
            legend={
                "orientation": "h",
                "xanchor": "left",
                "x": 0.0,
                "yanchor": "bottom",
                "y": 1.01,
                "itemsizing": "constant",
            },
            margin=PlotSettings().margins_figures[0],
            barmode="overlay",
            height=PlotSettings().heights_figures[0],
            autosize=True,
        )

        return (
            fig,
            {"display": "flex"},
            {"display": "block"},
            {"display": "flex", "align-items": "center", "justify-content": "center"},
            {"display": "flex", "padding-left": "25px"},
            {"display": "flex"},
        )


    @callback(
        Output("export-csv-hourly", "data"),
        Output("btn_csv_hourly", "n_clicks"),
        Input("btn_csv_hourly", "n_clicks"),
        Input("cams-switch-1", "on"),
        Input("threshold-slider", "value"),
        Input("input-data", "data"),
        prevent_initial_call=True,
    )
    def export_csv_hourly(
        n_clicks: int | None, cams_on: bool, selected_q: int, input_data: list[str]
    ) -> tuple[dict[str, str], int]:
        if (n_clicks == 0) or (n_clicks is None):
            raise PreventUpdate

        # Get data from cache
        cache_data = get_cached_data(engine, input_data)

        # Get flags
        test_data_flagged = reprocess_hourly(cache_data, cams_on, selected_q)

        # Create export data
        export, first_date, last_date = export_hourly(
            engine, cache_data, test_data_flagged, cams_on, selected_q
        )

        # Define filename for export file
        suffix = "_".join(
            [
                str(first_date)[:10].replace("-", ""),
                str(last_date)[:10].replace("-", ""),
                str(selected_q),
                "hourly.csv"
            ]
        )
        outfile = build_filename(engine, cache_data, suffix)

        return dict(content=export, filename=outfile), 0


    @app.callback(
        Output("collapse-1", "is_open"),
        [Input("info-button-1", "n_clicks")],
        [State("collapse-1", "is_open")],
    )
    def toggle_collapse_1(n: int, is_open: bool) -> bool:
        if n:
            return not is_open
        return is_open


    @callback(
        Output("graph-monthly", "figure"),
        Output("div-monthly", "style"),
        Output("div-text-2", "style"),
        Input("points-switch", "on"),
        Input("cams-switch-2", "on"),
        Input("trend-radio", "value"),
        Input("length-slider", "value"),
        Input("input-data", "data"),
    )
    def update_figure_2(
        points_on: bool,
        cams_on: bool,
        selected_trend: str,
        max_length: int,
        input_data: list[str] | None,
    ) -> tuple[go.Figure, dict[str, str], dict[str, str]]:
        if input_data == []:
            raise PreventUpdate

        if input_data is None:
            return empty_plot(""), {"display": "none"}, {"display": "none"}

        # Get data from cache
        cache_data = get_cached_data(engine, input_data)

        if cache_data == []:
            return empty_plot(""), {"display": "none"}, {"display": "none"}

        df_monplot = pd.read_json(cache_data.monthly_data_plot, orient="columns")
        y_pred_mon = pd.read_json(cache_data.cams_plus_mon, orient="index", typ="series")

        # Prepare data for monthly plot
        mtp = get_n_months_to_predict(df_monplot.index, cache_data.time_start)
        df_monplot_flagged = process_monthly(
            df_monplot, mtp, max_length, selected_trend, cache_data.par,
        )

        # Should CAMS data be plotted?
        use_cams = (
            cams_on &
            (get_cams_vars(cache_data.par) != "") &
            (cache_data.time_end >= ModelSettings().min_date_cams)
        )

        # Plot
        fig = go.Figure(layout=go.Layout(height=PlotSettings().heights_figures[1]))
        fig = plot_monthly(
            fig,
            df_monplot_flagged,
            y_pred_mon,
            cache_data.par,
            mtp,
            use_cams,
            points_on,
        )

        # Add logo Empa
        fig = add_logo(fig, 0.0, 0.0, 0.1, 0.2)

        # Add title, legend, margins
        meta = metadata[metadata["gaw_id"] == cache_data.cod]
        title = cache_data.par.upper() + " mole fraction (" + GawUnits[cache_data.par].value + ")"
        fig.update_layout(
            title={
                "text": meta["name"].item() + " (" + cache_data.cod + "): monthly means",
                "xanchor": "center",
                "x": 0.5,
            },
            legend={
                "orientation": "h",
                "xanchor": "left",
                "x": 0.0,
                "yanchor": "bottom",
                "y": 1.01,
                "traceorder": "reversed",
            },
            margin=PlotSettings().margins_figures[1],
            xaxis_title="Time UTC",
            yaxis_title=title,
            autosize=True,
        )

        return fig, {"display": "block"}, {"display": "flex"}


    @callback(
        Output("export-csv-monthly", "data"),
        Output("btn_csv_monthly", "n_clicks"),
        Input("btn_csv_monthly", "n_clicks"),
        Input("cams-switch-2", "on"),
        Input("trend-radio", "value"),
        Input("trend-radio", "options"),
        Input("length-slider", "value"),
        Input("input-data", "data"),
        prevent_initial_call=True,
    )
    def export_csv_monthly(
        n_clicks: int | None,
        cams_on: bool,
        selected_trend: str,
        labels_trend: list[dict[str, str]],
        max_length: int,
        input_data: list[str],
    ) -> tuple[dict[str, str], int]:
        if (n_clicks == 0) or (n_clicks is None):
            raise PreventUpdate

        # Get data from cache
        cache_data = get_cached_data(engine, input_data)

        # Extract the selected trend (dictionary) from the trend options (list)
        trend = [t for t in labels_trend if t["value"] == selected_trend][0]

        # Create export data
        export, first_date, last_date = export_monthly(
            engine, cache_data, cams_on, trend, max_length
        )

        # Define filename for export file
        suffix = "_".join(
            [
                str(first_date)[:10].replace("-", ""),
                str(last_date)[:10].replace("-", ""),
                str(max_length),
                "monthly.csv"
            ]
        )
        outfile = build_filename(engine, cache_data, suffix)

        return dict(content=export, filename=outfile), 0


    @app.callback(
        Output("collapse-2", "is_open"),
        [Input("info-button-2", "n_clicks")],
        [State("collapse-2", "is_open")],
    )
    def toggle_collapse_2(n: int, is_open: bool) -> bool:
        if n:
            return not is_open
        return is_open


    @callback(
        Output("graph-cycles", "figure"),
        Output("div-cycles", "style"),
        Output("div-text-3", "style"),
        Input("points-switch", "on"),
        Input("n-years", "value"),
        Input("input-data", "data"),
    )
    def update_figure_3(
        points_on: bool, n_years: int, input_data: list[str] | None
    ) -> tuple[go.Figure, dict[str, str], dict[str, str]]:
        if input_data == []:
            raise PreventUpdate

        if input_data is None:
            return empty_plot(""), {"display": "none"}, {"display": "none"}

        # Get data from cache
        cache_data = get_cached_data(engine, input_data)
        if cache_data == []:
            return empty_plot(""), {"display": "none"}, {"display": "none"}
        tz = list_to_class(input_data, UserInput).tz

        # Prepare data to plot
        plot_data = process_cycles(cache_data, n_years)

        # Define plot panels and line colors
        fig = make_subplots(
            rows=1,
            cols=3,
            horizontal_spacing=0.075,
            subplot_titles=("Diurnal cycle", "Seasonal cycle", "Variability cycle"),
        )

        # Plot diurnal cycle
        fig = plot_diurnal(
            fig,
            plot_data,
            cache_data.res,
            tz if cache_data.offset != 0 else "UTC",
            points_on,
            row=1,
            col=1,
        )

        # Plot seasonal cycle
        fig = plot_seasonal(
            fig,
            plot_data,
            points_on,
            row=1,
            col=2,
        )

        # Plot seasonal cycle of variability
        fig = plot_variability(
            fig,
            plot_data,
            cache_data.res,
            points_on,
            row=1,
            col=3,
        )

        # Add logo Empa
        fig = add_logo(fig, 0.36, 0.0, 0.075, 0.2)
        if cache_data.res == "hourly":
            fig = add_logo(fig, 0.0, 0.0, 0.075, 0.2)
            fig = add_logo(fig, 0.72, 0.0, 0.075, 0.2)

        # Add title, legend, margins
        meta = metadata[metadata["gaw_id"] == cache_data.cod]
        fig.update_layout(
            title={
                "text": meta["name"].item() + " (" + cache_data.cod + ")",
                "xanchor": "center",
                "x": 0.5,
                "yanchor": "top",
                "y": 0.98,
            },
            legend={"itemclick": False, "itemdoubleclick": False},
            margin=PlotSettings().margins_figures[2],
            hovermode="x unified",
            height=PlotSettings().heights_figures[2],
            autosize=True,
        )

        return fig, {"display": "block"}, {"display": "flex"}


    @callback(
        Output("export-csv-dc", "data"),
        Output("btn_csv_dc", "n_clicks"),
        Input("btn_csv_dc", "n_clicks"),
        Input("n-years", "value"),
        Input("input-data", "data"),
        prevent_initial_call=True,
    )
    def export_csv_dc(
        n_clicks: int | None, n_years: int, input_data: list[str]
    ) -> tuple[dict[str, str], int]:
        if (n_clicks == 0) or (n_clicks is None):
            raise PreventUpdate

        # Get data from cache
        cache_data = get_cached_data(engine, input_data)
        if cache_data.res == "monthly":
            raise PreventUpdate

        # Create export data
        export = export_diurnal(engine, cache_data, n_years)

        # Define filename for export
        suffix = "_".join(
            [
                str(cache_data.last_year - n_years),
                str(cache_data.last_year),
                "diurnal-cycle.csv"
            ]
        )
        outfile = build_filename(engine, cache_data, suffix)

        return dict(content=export, filename=outfile), 0


    @callback(
        Output("export-csv-sc", "data"),
        Output("btn_csv_sc", "n_clicks"),
        Input("btn_csv_sc", "n_clicks"),
        Input("n-years", "value"),
        Input("input-data", "data"),
        prevent_initial_call=True,
    )
    def export_csv_sc(
        n_clicks: int | None, n_years: int, input_data: list[str]
    ) -> tuple[dict[str, str], int]:
        if (n_clicks == 0) or (n_clicks is None):
            raise PreventUpdate

        # Get data from cache
        cache_data = get_cached_data(engine, input_data)

        # Create export data
        export = export_seasonal(engine, cache_data, n_years)

        # Define filename for export
        suffix = "_".join(
            [
                str(cache_data.last_year - n_years),
                str(cache_data.last_year),
                "seasonal-cycle.csv"
            ]
        )
        outfile = build_filename(engine, cache_data, suffix)

        return dict(content=export, filename=outfile), 0


    @callback(
        Output("export-csv-vc", "data"),
        Output("btn_csv_vc", "n_clicks"),
        Input("btn_csv_vc", "n_clicks"),
        Input("n-years", "value"),
        Input("input-data", "data"),
        prevent_initial_call=True,
    )
    def export_csv_vc(
        n_clicks: int | None, n_years: int, input_data: list[str]
    ) -> tuple[dict[str, str], int]:
        if (n_clicks == 0) or (n_clicks is None):
            raise PreventUpdate

        # Get data from cache
        cache_data = get_cached_data(engine, input_data)
        if cache_data.res == "monthly":
            raise PreventUpdate

        # Create export data
        export = export_variability(engine, cache_data, n_years)

        # Define filename for export
        suffix = "_".join(
            [
                str(cache_data.last_year - n_years),
                str(cache_data.last_year),
                "variability-cycle.csv"
            ]
        )
        outfile = build_filename(engine, cache_data, suffix)

        return dict(content=export, filename=outfile), 0


    @app.callback(
        Output("collapse-3", "is_open"),
        [Input("info-button-3", "n_clicks")],
        [State("collapse-3", "is_open")],
    )
    def toggle_collapse_3(n: int, is_open: bool) -> bool:
        if n:
            return not is_open
        return is_open


    return app



# Run app
if __name__ == "__main__":
    config = AppConfig()
    parser = ArgumentParser()
    parser.add_argument(
        "-p", "--port", default=8000, type=int, help="Port to listen to"
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Set debug mode")
    parser.add_argument(
        "--db", default="/data/gaw.db", type=Path, help="Path to the database file"
    )
    parser.add_argument("--cache", default="cache", help="Path to the cache directory")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as cachedir:
        logger.info(f"Cache in {cachedir}")
        cache = Cache(
            config={  # note that filesystem cache doesn't work on systems with ephemeral filesystems like Heroku
                "CACHE_TYPE": "filesystem",
                "CACHE_DIR": cachedir,
                "CACHE_THRESHOLD": 50,  # should be equal to maximum number of users on the app at a single time
            }
        )
        db_path: Path = args.db
        prefix = "//" if db_path.is_absolute() else "/"
        logger.info(f"Database path: {db_path.resolve()}")
        engine = create_engine(f"sqlite://{prefix}{str(db_path)}", pool_size=10)
        app = create_app(engine, cache, config.assets_path, config.theme, config.title)

        app.run_server(debug=args.debug, port=args.port)
