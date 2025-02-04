import logging

import plotly.graph_objects as go
from gaw_qc.log_utils.decorators import log_function

logger = logging.getLogger(__name__)


@log_function(logger)
def empty_plot(msg: str) -> go.Figure:
    """return an empty plot containing a text message
    :param msg: Message to print in the plot area
    :return: Figure object
    """
    fig = go.Figure()
    fig.update_layout(
        annotations=[
            {
                "text": msg,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 28},
            }
        ]
    )
    fig.update_xaxes(
        showticklabels=False,
        zeroline=False,
        showgrid=False,
        linecolor="black",
        mirror=True,
    )
    fig.update_yaxes(
        showticklabels=False,
        zeroline=False,
        showgrid=False,
        linecolor="black",
        mirror=True,
    )

    return fig


@log_function(logger)
def empty_subplot(fig: go.Figure, msg: str, row: int, col: int) -> go.Figure:
    """return an empty subplot containing a text message
    :param fig: Plotly figure object
    :param msg: Message to print in the plot area
    :param row: Row number
    :param col: Column number
    :return: Plotly figure object
    """
    fig.add_trace(
        go.Scatter(
            x=[1, 2, 3],
            y=[1, 2, 3],
            mode="markers+text",
            text=["", msg, ""],
            textfont_size=28,
            marker_opacity=0,
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, row=row, col=col)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=row, col=col)

    return fig
