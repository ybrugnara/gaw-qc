import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import dcc, html
from dash.development.base_component import Component
from gaw_qc.models.model_config import ModelSettings
from gaw_qc.pages.components.collapses import help_hourly, help_monthly, help_cycles
from gaw_qc.plotting.aesthetics import PlotSettings

msettings = ModelSettings()
psettings = PlotSettings()


def hourly_panel() -> Component:

    return html.Div(
        [
            # Description
            html.Div(
                [
                    html.P(
                        [
                            "The first panel shows time series of hourly means.",
                            " The yellow and red circles indicate anomalous outliers;",
                            " the shaded areas indicate periods of systematic biases",
                            " with respect to CAMS-derived predictions.",
                            " Click on the Help button for more information.",
                        ],
                    )
                ],
                id="div-text-1",
                className="text",
                style={"display": "none"},
            ),
            
            # Panel
            html.Div(
                [
                    # Graph
                    html.Div(
                        [
                            dcc.Loading(
                                dcc.Graph(id="graph-hourly"),
                                id="loading-1",
                                type="circle",
                            )
                        ]
                    ),
                    
                    # Control bar
                    html.Div(
                        [
                            # CAMS switch
                            html.Div(
                                [
                                    html.Label(
                                        "Toggle CAMS",
                                        style={
                                            "font-size": "14px",
                                            "padding-right": "5px",
                                        },
                                    ),
                                    daq.BooleanSwitch(
                                        id="cams-switch-1", on=True
                                    ),
                                ],
                                style={
                                    "display": "flex", 
                                    "padding-left": "10%",
                                },
                            ),
                            
                            # Strictness slider
                            html.Div(
                                [
                                    html.Label(
                                        "Strictness",
                                        style={"font-size": "14px"},
                                    ),
                                    daq.Slider(
                                        min=1,
                                        max=msettings.n_levels,
                                        step=1,
                                        value=2,
                                        marks={
                                            1: {
                                                "label": "More flags",
                                                "style": {"font-size": "14px"},
                                            },
                                            msettings.n_levels: {
                                                "label": "Fewer flags",
                                                "style": {"font-size": "14px"},
                                            },
                                        },
                                        id="threshold-slider",
                                        updatemode="mouseup",
                                        handleLabel={
                                            "showCurrentValue": True,
                                            "label": "level",
                                        },
                                        size=200,
                                    ),
                                ],
                                style={"padding-left": "10%"},
                            ),
                            
                            # Export button
                            html.Div(
                                [
                                    dbc.Button(
                                        "Export to CSV",
                                        id="btn_csv_hourly",
                                        #color="danger",
                                        size="sm",
                                        style={"background": psettings.colors_buttons[0]},
                                    ),
                                    dcc.Download(id="export-csv-hourly"),
                                ],
                                style={"padding-left": "10%"},
                            ),
                            
                            # Bin size slider
                            html.Div(
                                [
                                    html.Label(
                                        "Bin size", style={"font-size": "14px"}
                                    ),
                                    daq.Slider(
                                        min=0.1,
                                        max=10.0,
                                        step=0.1,
                                        value=1.0,
                                        marks={
                                            0.1: {
                                                "label": "0.1",
                                                "style": {"font-size": "14px"},
                                            },
                                            10: {
                                                "label": "10",
                                                "style": {"font-size": "14px"},
                                            },
                                        },
                                        id="bin-slider",
                                        size=200,
                                    ),
                                ],
                                style={"padding-left": "5%"},
                            ),
                            
                        ],
                        style={"display": "none"},
                        id="div-hourly-settings",
                    ),
                    
                    # Help button
                    html.Div(
                        [
                            dbc.Button(
                                "Help",
                                id="info-button-1",
                                #color="primary",
                                n_clicks=0,
                                size="sm",
                                style={"background": psettings.colors_buttons[1]},
                            ),
                            help_hourly(),
                        ],
                        style={"display": "none"},
                        id="div-hourly-info",
                    ),
                    
                ],
                className="panel",
                style={"display": "none"},
                id="div-hourly",
            ),
            
        ]
    )


def monthly_panel() -> Component:
    
    return html.Div(
        [
            # Description
            html.Div(
                [
                    html.P(
                        [
                            "The second panel shows time series of monthly means,",
                            " including a few years of historical data.",
                            " The red circles indicate anomalous outliers.",
                            " Click on the Help button for more information.",
                        ]
                    )
                ],
                id="div-text-2",
                className="text",
                style={"display": "none"},
            ),
            
            # Panel
            html.Div(
                [
                    html.Div(
                        [
                            # Graph
                            html.Div(
                                [
                                    dcc.Loading(
                                        dcc.Graph(id="graph-monthly"),
                                        id="loading-2",
                                        type="circle",
                                    )
                                ],
                                style={"width": "80%", "padding-left": "5%"},
                            ),
                            
                            # Control bar right
                            html.Div(
                                [
                                    # Trend type
                                    html.Label(
                                        "Type of trend",
                                        style={
                                            "font-size": "14px",
                                            "text-decoration": "underline",
                                            "margin-bottom": "10px",
                                        },
                                    ),
                                    dcc.RadioItems(
                                        options=[
                                            {"label": "No trend", "value": "n"},
                                            {"label": "Linear", "value": "c"},
                                            {
                                                "label": "Quadratic",
                                                "value": "t",
                                            },
                                        ],
                                        value="c",
                                        inline=False,
                                        id="trend-radio",
                                        labelStyle={"font-size": "14px"},
                                    ),
                                    
                                    # Years slider
                                    html.Label(
                                        "Number of years to use",
                                        style={
                                            "font-size": "14px",
                                            "text-decoration": "underline",
                                            "margin-top": "50px",
                                        },
                                    ),
                                    dcc.Slider(
                                        min=4,
                                        max=10,
                                        step=1,
                                        value=7,
                                        marks={
                                            i: {
                                                "label": str(i), 
                                                "style": {"font-size": "16px"}
                                                } for i in range(4, 11)
                                            },
                                        id="length-slider",
                                        updatemode="mouseup",
                                        vertical=True,
                                        verticalHeight=200,
                                    ),
                                    
                                ],
                                style={
                                    "width": "15%",
                                    "padding-top": "100px",
                                },
                            ),
                            
                        ],
                        style={
                            "display": "flex",
                            "margin-left": "auto",
                            "margin-right": "auto",
                        },
                    ),
                    
                    # Control bar bottom
                    html.Div(
                        [
                            # CAMS switch
                            html.Div(
                                [
                                    html.Label(
                                        "Toggle CAMS",
                                        style={
                                            "font-size": "14px",
                                            "padding-right": "5px",
                                        },
                                    ),
                                    daq.BooleanSwitch(
                                        id="cams-switch-2", on=True
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "padding-left": "20%",
                                },
                            ),
                            
                            # Export button
                            html.Div(
                                [
                                    dbc.Button(
                                        "Export to CSV",
                                        id="btn_csv_monthly",
                                        #color="danger",
                                        size="sm",
                                        style={"background": psettings.colors_buttons[0]},
                                    ),
                                    dcc.Download(id="export-csv-monthly"),
                                ],
                                style={"padding-left": "40%"},
                            ),
                            
                        ],
                        style={"display": "flex"},
                    ),
                    
                    # Help button
                    html.Div(
                        [
                            dbc.Button(
                                "Help",
                                id="info-button-2",
                                #color="primary",
                                n_clicks=0,
                                size="sm",
                                style={"background": psettings.colors_buttons[1]},
                            ),
                            help_monthly(),
                        ],
                        style={"display": "flex", "padding-left": "25px"},
                    ),
                    
                ],
                className="panel",
                style={"display": "none"},
                id="div-monthly",
            ),
            
        ]
    )


def cycles_panel() -> Component:

    return html.Div(
        [
            # Description
            html.Div(
                [
                    html.P(
                        [
                            "The third panel shows comparisons with other years",
                            " for diurnal, seasonal, and variability cycle.",
                            " The analyzed data is represented by black dots.",
                            " Click on the Help button for more information.",
                        ]
                    )
                ],
                id="div-text-3",
                className="text",
                style={"display": "none"},
            ),
            
            # Panel
            html.Div(
                [
                    # Graph
                    html.Div(
                        [
                            dcc.Loading(
                                dcc.Graph(id="graph-cycles"),
                                id="loading-3",
                                type="circle",
                            )
                        ]
                    ),
                    
                    # Control bar
                    html.Div(
                        [
                            # Years slider
                            html.Div(
                                [
                                    html.Label(
                                        "Number of years to use",
                                        style={"font-size": "14px"},
                                    ),
                                    dcc.Slider(
                                        min=1,
                                        max=7,
                                        step=1,
                                        value=4,
                                        marks={
                                            i: {
                                                "label": str(i), 
                                                "style": {"font-size": "16px"}
                                                } for i in range(1, 8)
                                            },
                                        id="n-years",
                                        updatemode="mouseup",
                                    ),
                                ],
                                style={
                                    "padding-left": "5%",
                                    "width": "20%",
                                },
                            ),
                            
                            # Export buttons
                            html.Div(
                                [
                                    dbc.Button(
                                        "Export to CSV",
                                        id="btn_csv_dc",
                                        #color="danger",
                                        size="sm",
                                        style={"background": psettings.colors_buttons[0]},
                                    ),
                                    dcc.Download(id="export-csv-dc"),
                                ],
                                style={"padding-left": "2%"},
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        "Export to CSV",
                                        id="btn_csv_sc",
                                        #color="danger",
                                        size="sm",
                                        style={"background": psettings.colors_buttons[0]},
                                    ),
                                    dcc.Download(id="export-csv-sc"),
                                ],
                                style={"padding-left": "22%"},
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        "Export to CSV",
                                        id="btn_csv_vc",
                                        #color="danger",
                                        size="sm",
                                        style={"background": psettings.colors_buttons[0]},
                                    ),
                                    dcc.Download(id="export-csv-vc"),
                                ],
                                style={"padding-left": "22%"},
                            ),
                            
                        ],
                        style={"display": "flex"},
                    ),
                    
                    # Help button
                    html.Div(
                        [
                            dbc.Button(
                                "Help",
                                id="info-button-3",
                                #color="primary",
                                n_clicks=0,
                                size="sm",
                                style={"background": psettings.colors_buttons[1]},
                            ),
                            help_cycles(),
                        ],
                        style={"display": "flex", "padding-left": "25px"},
                    ),
                ],
                className="panel",
                style={"display": "none"},
                id="div-cycles",
            ),
            
        ]
    )
