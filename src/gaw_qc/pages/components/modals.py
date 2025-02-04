import dash_bootstrap_components as dbc
from dash import html


def open_acknowledgments() -> dbc.Modal:

    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Acknowledgments")),
            dbc.ModalBody(
                [
                    html.P(
                        [
                            "This app was developed by the ",
                            html.A(
                                "Quality Assurance / Scientific Activity Centre (QA/SAC) Switzerland",
                                href="https://www.empa.ch/web/s503/qa-sac-switzerland",
                                target="_blank",
                            ),
                            " at Empa, which is supported by the Federal Office of Meteorology MeteoSwiss",
                            " in the framework of the Global Atmosphere Watch Programme of the World Meteorological Organization.",
                            " Technical support was provided by the Scientific IT department at Empa.",
                        ]
                    ),
                    html.P(
                        [
                            "Use of the output is free within the limits of the underlying data policies; ",
                            "in any case, appropriate credit must be given to QA/SAC Switzerland.",
                        ]
                    ),
                    html.P(
                        [
                            "For additional information please contact ",
                            html.A(
                                "gaw@empa.ch",
                                href="mailto:gaw@empa.ch",
                            ),
                        ]
                    ),
                ]
            ),
        ],
        id="modal-acknowl",
        size="lg",
        is_open=False,
    )


def open_credits() -> dbc.Modal:

    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Data credits")),
            dbc.ModalBody(
                [
                    html.P(
                        [
                            html.B(
                                [
                                    "Measurements of CH",
                                    html.Sub("4"),
                                    ", CO, and CO",
                                    html.Sub("2"),
                                ]
                            ),
                            " are provided by the ",
                            html.B("World Data Centre for Greenhouse Gases (WDCGG) "),
                            "at JMA. Data users must follow the ",
                            html.A(
                                "GAW data policy",
                                href="https://gaw.kishou.go.jp/policy/gaw",
                                target="_blank",
                            ),
                            ". Note that the download of some data series require",
                            " logging in on the WDCGG website; for those series"
                            " the export buttons are not available."
                        ]
                    ),
                    html.P(
                        [
                            html.B(
                                [
                                    "Measurements of O",
                                    html.Sub("3"),
                                ]
                            ),
                            " are provided by the ",
                            html.B("World Data Centre for Reactive Gases (WDCRG) "),
                            "at NILU. The data is licensed under a ",
                            html.A(
                                "Creative Commons Attribution 4.0 International License",
                                href="http://creativecommons.org/licenses/by/4.0/",
                                target="_blank",
                            ),
                            ".",
                        ]
                    ),
                    html.P(
                        [
                            "Some of the plots are generated using ",
                            html.A(
                                "Copernicus Atmosphere Monitoring Service",
                                href="https://atmosphere.copernicus.eu/",
                                target="_blank",
                            ),
                            " information and/or contain modified Copernicus ",
                            "Atmosphere Monitoring Service information [2024].",
                        ]
                    ),
                ]
            ),
        ],
        id="modal-credits",
        size="lg",
        is_open=False,
    )


def open_report_bugs() -> dbc.Modal:

    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Report bug")),
            dbc.ModalBody(
                [
                    html.P(
                        [
                            "Please report any bug to ",
                            html.A(
                                "yuri.brugnara@empa.ch",
                                href="mailto:yuri.brugnara@empa.ch",
                            ),
                            ", specifying station, variable, and period affected. ",
                            "If uploaded data is analyzed, please also send the data file if possible.",
                        ]
                    ),
                ]
            ),
        ],
        id="modal-report-bug",
        size="lg",
        is_open=False,
    )
