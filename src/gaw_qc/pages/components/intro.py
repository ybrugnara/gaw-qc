import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.development.base_component import Component
from gaw_qc.pages.components.modals import (
    open_acknowledgments, open_credits, open_report_bugs
)
from gaw_qc.plotting.aesthetics import PlotSettings


def intro(
        assets_path: str,
        n_stations: int,
        last_update_cams: str,
        last_update_gaw: str
) -> Component:
    bcolor = PlotSettings().colors_buttons[1]

    return html.Div(
        [
            html.Div(
                [
                    # Introduction
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.P(html.H3("Welcome to GAW-QC!")),
                                    html.P(
                                        [
                                            "On this page you can visualize the measurements of ",
                                            html.B("methane, carbon dioxide, carbon monoxide, and ozone"),
                                            " that are available in the database of the ",
                                            html.B("Global Atmosphere Watch (GAW)"),
                                            " World Data Centres, and compare them with model data by the ",
                                            "Copernicus Atmosphere Monitoring Service (CAMS; from 2020 onward).",
                                        ]
                                    ),
                                    html.P(
                                        [
                                            html.B("You can also upload recent measurements"),
                                            " from a GAW station to check their quality. ",
                                            "Any anomaly in the measurements will be highlighted by a data-driven algorithm ",
                                            "based on historical and CAMS data."
                                        ]
                                    ),
                                    html.P(
                                        [
                                            "For more information see the ",
                                            html.A(
                                                "wiki",
                                                href="https://github.com/ybrugnara/gaw-qc/wiki",
                                                target="_blank",
                                            ),
                                            ".",
                                        ]
                                    ),

                                    # Buttons for modals
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "Data credits",
                                                        id="open-credits",
                                                        n_clicks=0,
                                                        #color="primary",
                                                        size="sm",
                                                        style={"background": bcolor},
                                                    ),
                                                    open_credits(),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "Acknowledgments",
                                                        id="open-acknowl",
                                                        n_clicks=0,
                                                        #color="primary",
                                                        size="sm",
                                                        style={"background": bcolor},
                                                    ),
                                                    open_acknowledgments(),
                                                ],
                                                style={"padding-left": "25px"},
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Button(
                                                        "Report bug",
                                                        id="open-report-bug",
                                                        n_clicks=0,
                                                        #color="primary",
                                                        size="sm",
                                                        style={"background": bcolor},
                                                    ),
                                                    open_report_bugs(),
                                                ],
                                                style={"padding-left": "25px"},
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "padding-top": "25px",
                                            #"align-items": "center",
                                            #"justify-content": "center",
                                        },
                                    ),
                                ],
                            ),
                        ],
                        style={
                            "font-size": "18px",
                            "width": "50%",
                            "text-align": "justify",
                            "text-justify": "inter-word",
                            "padding-left": "25px",
                            "padding-right": "50px",
                            "border-right": "1px solid #cccccc",
                        },
                    ),

                    # Map of stations
                    html.Div(
                        [
                            html.Div(
                                html.P(
                                    [
                                        "Currently ",
                                        html.B(
                                            str(n_stations) + " GAW stations"
                                        ),
                                        " supported",
                                    ]
                                ),
                                style={
                                    "text-align": "center",
                                    "font-size": "16px",
                                    "padding-bottom": "10px",
                                    "padding-top": "10px",
                                },
                            ),
                            html.Div(
                                dcc.Graph(
                                    id="map-of-stations",
                                    style={"width": "500px"},
                                ),
                                style={
                                    "display": "flex",
                                    "justify-content": "center",
                                    "align-items": "center",
                                },
                            ),
                            html.Div(
                                html.P(
                                    [
                                        "Date of latest GAW data point: ",
                                        html.B(last_update_gaw),
                                        html.Br(),
                                        "Date of latest CAMS data point: ",
                                        html.B(last_update_cams),
                                    ]
                                ),
                                style={
                                    "text-align": "center",
                                    "font-size": "16px",
                                    "padding-top": "25px",
                                },
                            ),
                        ],
                        style={"width": "50%"},
                    ),
                ],
                style={"display": "flex"},
            ),

        ],
        className="intro",
    )
