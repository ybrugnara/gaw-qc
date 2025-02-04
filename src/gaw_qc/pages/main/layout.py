from dash import dcc, html
from dash.development.base_component import Component
from gaw_qc.config.app_config import AppConfig
from gaw_qc.pages.components.header import header
from gaw_qc.pages.components.intro import intro
from gaw_qc.pages.components.input_form import input_form, lp_switch
from gaw_qc.pages.components.panels import hourly_panel, monthly_panel, cycles_panel

assets_path = AppConfig().assets_path.as_posix()


def main_layout(
        stations: list[str],
        last_update_cams: str,
        last_update_gaw: str
) -> Component:

    return html.Div(
        [

            html.Div(
                [
                    # Header
                    header(AppConfig().assets_path.as_posix()),

                    # Intro
                    html.Div(
                        intro(
                            AppConfig().assets_path.as_posix(),
                            len(stations),
                            last_update_cams,
                            last_update_gaw
                        ),
                    ),

                    # Input form
                    html.Div(
                        input_form(stations),
                        style={
                            "background": "#eeeeee",
                            "padding-bottom": "25px",
                            "padding-top": "25px",
                        },
                    ),
                ],
                style={
                    "border-style": "solid",
                    "border-color": "#cccccc",
                    "border-width": "1px",
                }

            ),

            # Data loading with message
            html.Div(
                [
                    html.Label([], id="label-wait"),

                    # Store input parameters (also used to signal callbacks when to fire)
                    dcc.Loading(
                        id="loading-data",
                        type="circle",
                        fullscreen=False,
                        children=dcc.Store(
                            id="input-data", data=[], storage_type="memory"
                        ),
                    ),
                ],
                style={"text-align": "center", "margin-top": "25px"},
            ),
            
            # Add lines/points switch
            html.Div(
                lp_switch(),
            ),
            
            # Dashboard
            html.Div(
                [
                    hourly_panel(),
                    monthly_panel(),
                    cycles_panel(),
                ],
            ),

        ],        
        className="layout",
    )
