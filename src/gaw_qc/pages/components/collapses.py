import dash_bootstrap_components as dbc
from dash import html
from gaw_qc.models.model_config import ModelSettings
from gaw_qc.plotting.aesthetics import PlotSettings

msettings = ModelSettings()
psettings = PlotSettings()


def help_hourly() -> dbc.Collapse:
    
    return dbc.Collapse(
        dbc.Card(
            dbc.CardBody(
                [
                    html.P(
                        [
                            html.B(psettings.labels_plot[0]),
                            html.Br(),
                            "Hourly means of the measurements made at the station for the selected / uploaded period.",
                        ]
                    ),
                    html.P(
                        [
                            html.B(psettings.labels_plot[1]),
                            html.Br(),
                            "Data from the Copernicus Atmosphere Monitoring Service ",
                            html.A(
                                "global atmospheric composition forecasts",
                                href="https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts?tab=overview",
                                target="_blank",
                            ),
                            " or the ",
                            html.A(
                                "global greenhouse gas forecasts",
                                href="https://ads.atmosphere.copernicus.eu/datasets/cams-global-greenhouse-gas-forecasts?tab=overview",
                                target="_blank",
                            ),
                            " (for CO", html.Sub("2"), "),"
                            " two numerical model simulations that assimilate satellite measurements.",
                            " This product can have a large bias and is given just for reference.",
                            " The original 3-hourly resolution was increased to hourly through linear interpolation.",
                            " Note that it is not available before 2020",
                            " (before March 2024 for CO", html.Sub("2"), ")",
                            " and for N", html.Sub("2"), "O.",
                        ]
                    ),
                    html.P(
                        [
                            html.B(psettings.labels_plot[2]),
                            html.Br(),
                            "A statistically improved version of ",
                            psettings.labels_plot[1],
                            " in which the data are debiased through machine learning (ML).",
                            " Periods with significant deviations from the measurements are highlighted by a ",
                            html.B("yellow shading"),
                            " and indicate a potential bias in the measurements.",
                            " However, the ML model has limitations and does not always",
                            " deliver reliable results.",
                            " Some expertise by the user is required for a correct interpretation.",
                            " This product is only available if at least ",
                            str(msettings.min_months_ml),
                            " months of overlap exist between CAMS and station data.",
                        ]
                    ),
                    html.P(
                        [
                            html.B(psettings.labels_plot[3]),
                            html.Br(),
                            "Data that exhibit an anomalous behaviour with respect to historical data.",
                            " The flags are based on the anomaly score of the Local Outlier Factor (LOF)",
                            " method applied to sequences of ",
                            str(msettings.window_size), " measurements.",
                        ]
                    ),
                    html.P(
                        [
                            html.B(psettings.labels_plot[4]),
                            html.Br(),
                            "Data that exhibit a very anomalous behaviour with respect",
                            " to historical data (see yellow flags),",
                            " or yellow flags that occur in a period where measurements",
                            " are significantly different from ",
                            psettings.labels_plot[2], " (i.e., double yellow flags).",
                        ]
                    ),
                    html.P(
                        [
                            html.B("Toggle CAMS"),
                            html.Br(),
                            "Remove ", psettings.labels_plot[1], " and ",
                            psettings.labels_plot[2],
                            " time series from the plot and from the export file.",
                            " To just hide any time series you can click on the",
                            " respective entry in the legend (they will still be exported).",
                        ]
                    ),
                    html.P(
                        [
                            html.B("Strictness"),
                            html.Br(),
                            "The strictness slider allows you to adjust the significance levels",
                            " required to produce yellow and red flags.",
                            " A higher strictness level implies higher probability",
                            " thresholds and fewer flags.",
                            " This is useful if you think that the algorithm is flagging",
                            " too many measurements or vice versa.",
                        ]
                    ),
                    html.P(
                        [
                            html.B("Pie chart"),
                            html.Br(),
                            "The chart on the top-right corner shows the fraction of normal (green),",
                            " anomalous (yellow), and very anomalous (red) data.",
                            " Both yellow flags and yellow shadings are considered anomalous data.",
                        ]
                    ),
                    html.P(
                        [
                            html.B("Histogram"),
                            html.Br(),
                            "The chart on the bottom-right corner shows the distribution of the ",
                            psettings.labels_plot[0], ", ",
                            psettings.labels_plot[1], ", and ",
                            psettings.labels_plot[2], ",",
                            " using the same color code of the main chart.",
                            " It is possible to adjust the size of the bins through a slider.",
                        ]
                    ),
                    html.P(
                        [
                            html.B("Export to CSV"),
                            html.Br(),
                            "Download the time series shown in the main chart (flags included)",
                            " as a Comma Separated Value (CSV) file.",
                            " This function may not be available for some stations",
                            " due to data policy limitations.",
                        ]
                    ),
                ]
            ),
            style={"width": psettings.width_collapse},
        ),
        id="collapse-1",
        is_open=False,
    )


def help_monthly() -> dbc.Collapse:
    
    return dbc.Collapse(
        dbc.Card(
            dbc.CardBody(
                [
                    html.P(
                        [
                            html.B(psettings.labels_plot[0]),
                            html.Br(),
                            "Monthly means of the measurements made at the station in recent years",
                            " (the number of years is controlled by the slider next to the chart).",
                            " Months with less than ",
                            str(msettings.n_min),
                            " valid hourly values are not shown.",
                        ]
                    ),
                    html.P(
                        [
                            html.B(psettings.labels_plot[1]),
                            html.Br(),
                            "Data from the Copernicus Atmosphere Monitoring Service ",
                            html.A(
                                "global atmospheric composition forecasts",
                                href="https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts?tab=overview",
                                target="_blank",
                            ),
                            " or the ",
                            html.A(
                                "global greenhouse gas forecasts",
                                href="https://ads.atmosphere.copernicus.eu/datasets/cams-global-greenhouse-gas-forecasts?tab=overview",
                                target="_blank",
                            ),
                            " (for CO", html.Sub("2"), "),"
                            " two numerical model simulations that assimilate satellite measurements.",
                            " This product can have a large bias and is given just for reference.",
                            " The original 3-hourly resolution was increased to hourly through linear interpolation.",
                            " Note that it is not available before 2020",
                            " (before March 2024 for CO", html.Sub("2"), ")",
                            " and for N", html.Sub("2"), "O.",
                        ]
                    ),
                    html.P(
                        [
                            html.B(psettings.labels_plot[2]),
                            html.Br(),
                            "A statistically improved version of CAMS in which the",
                            " data are debiased through machine learning (ML).",
                            " A large difference with the measurements might indicate",
                            " a systematic bias.",
                            " However, the ML model has limitations and does not always",
                            " deliver reliable results.",
                            " Some expertise by the user is required for a correct interpretation.",
                            " This product is only available if at least ",
                            str(msettings.min_months_ml),
                            " months of overlap exist between CAMS and station data.",
                        ]
                    ),
                    html.P(
                        [
                            html.B(psettings.labels_plot[5]),
                            html.Br(),
                            "Predicted values for the selected / uploaded period using",
                            " a Seasonal Auto-Regressive Integrated Moving Average model",
                            " based on the historical data, with ",
                            str(int(100 * (1 - msettings.p_conf))),
                            "% confidence range. ",
                            psettings.labels_plot[5],
                            " is a purely statistical alternative to ",
                            psettings.labels_plot[2], ".",
                            " Only the data shown in the plot are considered in order to fit the model,",
                            " meaning that the prediction can change by modifying the number of years.",
                            " Moreover, the",
                            html.B(" type of trend"),
                            " assumed by the model can be selected by the user (top-right corner).",
                            html.Br(),
                            "The ", psettings.labels_plot[5], " prediction will",
                            " not be shown if less than ", str(msettings.min_months_sarima),
                            " months of past data are available.",
                        ]
                    ),
                    html.P(
                        [
                            html.B("Flags"),
                            html.Br(),
                            "Measurements that are outside the confidence range of the ",
                            psettings.labels_plot[5],
                            " prediction are marked with a red circle.",
                        ]
                    ),
                    html.P(
                        [
                            html.B("Toggle CAMS"),
                            html.Br(),
                            "Remove ", psettings.labels_plot[1], " and ",
                            psettings.labels_plot[2],
                            " time series from the plot and from the export file.",
                            " To just hide any time series you can click on the",
                            " respective entry in the legend (they will still be exported).",
                        ]
                    ),
                    html.P(
                        [
                            html.B("Export to CSV"),
                            html.Br(),
                            "Download the data from the selected / uploaded period (flags included)",
                            " as a Comma Separated Value (CSV) file.",
                            " This function may not be available for some stations",
                            " due to data policy limitations.",
                        ]
                    ),
                ]
            ),
            style={"width": psettings.width_collapse},
        ),
        id="collapse-2",
        is_open=False,
    )


def help_cycles() -> dbc.Collapse:
    
    return dbc.Collapse(
        dbc.Card(
            dbc.CardBody(
                [
                    html.P(
                        [
                            html.B("Diurnal cycle"),
                            html.Br(),
                            "The first panel compares the mean diurnal cycle in",
                            " the selected / uploaded period with other years and",
                            " with the multi-year average.",
                            " If there is a trend, the curves will be shifted accordingly.",
                            " However, the shape of the curves should be fairly similar.",
                            " If not, it might be worth investigating the causes",
                            " of the differences.",
                            html.Br(),
                            "By hovering over a certain value the number of underlying",
                            " hourly values is also displayed.",
                        ]
                    ),
                    html.P(
                        [
                            html.B("Seasonal cycle"),
                            html.Br(),
                            "The second panel compares the monthly mean values in",
                            " the selected / uploaded period with other years and",
                            " with the multi-year average.",
                            " If there is a trend, the curves will be shifted accordingly.",
                            " However, the shape of the curves should be fairly similar.",
                            " If not, it might be worth investigating the causes",
                            " of the differences.",
                            html.Br(),
                            "Months with less than ",
                            str(msettings.n_min),
                            " valid hourly values are not shown.",
                        ]
                    ),
                    html.P(
                        [
                            html.B("Variability cycle"),
                            html.Br(),
                            "The third panel compares the monthly variability in",
                            " the selected / uploaded period with other years and",
                            " with the multi-year average.",
                            " The variability is defined as the standard deviation",
                            " of the hourly means. All curves should be fairly similar.",
                            " If not, it might be worth investigating the causes",
                            " of the differences.",
                            html.Br(),
                            "Months with less than ",
                            str(msettings.n_min),
                            " valid hourly values are not shown.",
                        ]
                    ),
                    html.P(
                        [
                            html.B("Export to CSV"),
                            html.Br(),
                            "Download the data shown in a specific panel as a",
                            " Comma Separated Value (CSV) file.",
                            " This function may not be available for some stations",
                            " due to data policy limitations.",
                        ]
                    ),
                ]
            ),
            style={"width": psettings.width_collapse},
        ),
        id="collapse-3",
        is_open=False,
    )
