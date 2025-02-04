import dash_daq as daq
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.development.base_component import Component
from gaw_qc.plotting.aesthetics import PlotSettings


def input_form(stations: list[str]) -> Component:
    psettings = PlotSettings()

    return html.Div(
        [
            html.Div(
                [html.Label(html.B("1. Select the station, gas, and inlet height"))],
                style={"text-align": "center", "font-size": "20px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Dropdown(
                                stations,
                                id="station-dropdown",
                                placeholder="Select station",
                                style={"font-size": "16px"},
                            )
                        ],
                        style={"width": "200px"},
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="param-dropdown",
                                placeholder="Select parameter",
                                style={"font-size": "16px"},
                            )
                        ],
                        style={"width": "200px", "padding-left": "25px"},
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="height-dropdown",
                                placeholder="Select height (m)",
                                style={"font-size": "16px"},
                            )
                        ],
                        style={"width": "200px", "padding-left": "25px"},
                    ),
                    dcc.Store(
                        id="stat-par-hei",
                        data=[],
                        storage_type="memory",
                    ),
                ],
                style={
                    "display": "flex",
                    "margin-top": "25px",
                    "align-items": "center",
                    "justify-content": "center",
                },
            ),
            html.Div(
                [
                    html.Label([], id="label-contributor"),
                ],
                style={
                    "font-size": "16px",
                    "margin-top": "10px",
                    "text-align": "center"},
            ),
            html.Div(
                [
                    html.P(
                        html.B(
                            "2. Select a period to analyze or upload your own data (hourly means)"
                        )
                    )
                ],
                style={
                    "font-size": "20px",
                    "margin-top": "25px",
                    "text-align": "center",
                },
            ),
            html.Div(
                [
                    dcc.Loading(
                        id="loading-dates",
                        type="circle",
                        fullscreen=False,
                        children=dcc.DatePickerRange(
                            id="date-range", updatemode="bothdates"
                        ),
                    )
                ],
                style={
                    "display": "flex",
                    "margin-top": "25px",
                    "align-items": "center",
                    "justify-content": "center",
                },
            ),
            html.Div(
                [
                    html.Label(
                        "(a maximum period length of one year can be selected)"
                    )
                ],
                style={
                    "text-align": "center",
                    "margin-top": "10px",
                    "font-size": "12px",
                },
            ),
            html.Div(
                [html.Label("or", style={"font-size": "18px"})],
                style={
                    "display": "flex",
                    "margin-top": "25px",
                    "align-items": "center",
                    "justify-content": "center",
                },
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        [
                            "UTC",
                            "UTC-11:00",
                            "UTC-10:00",
                            "UTC-09:00",
                            "UTC-08:00",
                            "UTC-07:00",
                            "UTC-06:00",
                            "UTC-05:00",
                            "UTC-04:00",
                            "UTC-03:00",
                            "UTC-02:00",
                            "UTC-01:00",
                            "UTC+01:00",
                            "UTC+02:00",
                            "UTC+03:00",
                            "UTC+04:00",
                            "UTC+05:00",
                            "UTC+06:00",
                            "UTC+07:00",
                            "UTC+08:00",
                            "UTC+09:00",
                            "UTC+10:00",
                            "UTC+11:00",
                            "UTC+12:00",
                        ],
                        id="timezone-dropdown",
                        placeholder="Choose a time zone",
                        style={
                            "font-size": "16px",
                            "width": "250px",
                            "heigth": "50px",
                        },
                    ),
                    html.Label(
                        "and",
                        style={"font-size": "18px", "padding-left": "25px"},
                    ),
                    dcc.Upload(
                        children=dbc.Button(
                            "Upload File (.csv/.xls)*",
                            #color="danger",
                            style={
                                "font-size": "18px",
                                "width": "250px",
                                "height": "50px",
                                "margin-left": "25px",
                                "background": psettings.colors_buttons[0],
                            },
                        ),
                        id="upload-data",
                    ),
                ],
                style={
                    "display": "flex",
                    "margin-top": "25px",
                    "align-items": "center",
                    "justify-content": "center",
                },
            ),
            html.Div(
                [
                    html.Label(
                        "*The file must have two columns: time and value (hourly mean). It cannot exceed one year of data. An example is given"
                    ),
                    html.A(
                        "here",
                        target="_blank",
                        href="https://raw.githubusercontent.com/ybrugnara/gaw-qc/main/examples/Jungfraujoch_CO2_20240101-20240331.csv",
                        style={"padding-left": "3px"},
                    ),
                ],
                style={
                    "text-align": "center",
                    "margin-top": "10px",
                    "font-size": "12px",
                },
            ),
        ],
        className="menu",
    )


def lp_switch() -> Component:
    
    return html.Div(
        [
            html.Label(
                "Lines",
                style={
                    "font-size": "14px",
                    "font-weight": "bold",
                    "margin-top": "3px",
                },
            ),
            daq.BooleanSwitch(
                id="points-switch",
                on=False,
                color="#e7e7e7",
                style={"padding-left": "5px"},
            ),
            html.Label(
                "Points",
                style={
                    "font-size": "14px",
                    "font-weight": "bold",
                    "padding-left": "5px",
                    "margin-top": "3px",
                },
            ),
        ],
        className="text",
        style={"display": "none"},
        id="div-switch",
    )