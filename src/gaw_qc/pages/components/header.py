from dash import html
from dash.development.base_component import Component


def header(assets_path: str) -> Component:

    return html.Div(
        [
            html.Img(
                src=assets_path + "/logos/Logo_Empa.png",
                width="250px",
                className="left-image",
            ),
            html.Img(
                src=assets_path + "/logos/gaw-qc_logo.png",
                className="center-image",
            ),
            html.Img(
                src=assets_path + "/logos/wmo-gaw.png",
                width="250px",
                className="right-image",
            ),
        ],
        style={
            "padding-bottom": "5px",
            "padding-top": "5px",
        },
        className="header"
    )
