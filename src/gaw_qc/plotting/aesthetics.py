
import plotly.graph_objects as go
from pydantic_settings import BaseSettings
from numpy import linspace
from matplotlib.pyplot import get_cmap
from matplotlib.colors import rgb2hex
from gaw_qc.config.app_config import AppConfig
from gaw_qc.models.model_config import ModelSettings


def generate_palette(pname: str, n: int) -> list[str]:
    """Extract n evenly spacedcolors from a palette with increasingly darker
    shades after excluding too light shades
    :param pname: name of matplotlib palette
    :param n: number of colors in palette
    :return: list of colors as hexadecimals
    """
    cmap = get_cmap(pname)
    values = linspace(0.05, 1, n)
    return [rgb2hex(cmap(value)) for value in values]


class PlotSettings(BaseSettings):
    """Graphical parameters for the plots
    (NB: the width of the plots is set in the css file in the assets)
    color_bias: color to highlight periods with bias
    colors_buttons: colors used for large and small buttons
    colors_c: colors used for target year, multi-year mean in the cycle plots
    colors_h: colors used for measurements, CAMS, CAMS+ in the hourly plot
    colors_m: colors used for measurements, SARIMA, CAMS, CAMS+ in the monthly plot
    colors_map: colors used for all stations and the selected station in the map
    colors_pie: colors used for the flags and pie chart
    colors_years: color palette for third panel
    heights_figures: heights in pixels of the main plots
    labels_pie: labels for green, yellow, red slices of pie chart
    labels_plot: labels for measurements, CAMS, CAMS+, yellow flags, red flags, SARIMA
    line_widths_c: line widths for target year, multi-year mean, other years in the cycle plots
    line_widths_h: line widths for measurements, CAMS, CAMS+ in the hourly plot
    line_widths_m: line widths for measurements, SARIMA in the monthly plot
    margins_figures: margins for each figure in pixels
    marker_flag_h: arguments for hourly flags (colors are set by colors_pie)
    marker_flag_m: arguments for monthly flags
    marker_sizes_c: marker sizes for target year, multi-year mean, other years in the cycle plots
    marker_sizes_h: marker sizes for measurements, CAMS, CAMS+ in the hourly plot
    marker_sizes_m: marker sizes for measurements, SARIMA, CAMS, CAMS+ in the monthly plot
    marker_sizes_map: marker sizes for points in the map
    marker_symbols_m: marker symbols for measurements, SARIMA, CAMS, CAMS+ in the monthly plot
    months: abbreviations for x axis
    opacity_bias: opacity of the area that highlights periods with bias
    opacity_hist: opacity histograms
    x_args: arguments for all x axes
    y_args: arguments for all y axes
    width_collapse: width of the help collapse boxes
    """

    color_bias: str = "yellow"

    colors_buttons: list[str] = ["royalblue", "firebrick"]

    colors_c: list[str] = ["black", "dimgray"]

    colors_h: list[str] = ["black", "silver", "royalblue"]

    colors_m: list[str] = ["black", "skyblue", "silver", "royalblue"]

    colors_map: list[str] = ["royalblue", "red"]

    colors_pie: list[str] = ["green", "yellow", "red"]

    colors_years: list[str] = generate_palette(
        "Blues", ModelSettings().n_years_max + 2
    )
        #"#f0f921",
        #"#fdb42f",
        #"#ed7953",
        #"#cc4778",
        #"#9c179e",
        #"#5c01a6",
        #"#0d0887",
    #]

    heights_figures: list[int] = [500, 500, 500]

    labels_pie: list[str] = [
        "normal",
        "anomalous",
        "very anomalous",
    ]

    labels_plot: list[str] = [
        "Measurements",
        "CAMS",
        "CAMS+",
        "Yellow flags (LOF)",
        "Red flags (LOF)",
        "SARIMA",
    ]

    line_widths_c: list[float] = [0., 3., 1.]

    line_widths_h: list[float] = [2., .75, 1.5]

    line_widths_m: list[float] = [1.5, 1.5]

    margins_figures: list[dict[str, int]] = [
        dict(t=75), # figure 1 (hourly)
        dict(t=75), # figure 2 (monthly)
        dict(t=75), # figure 3 (cycles)
    ]
    
    marker_flag_h: dict[str, str|float|dict[str,float]] = dict(
        size=8., symbol="circle-open", line=dict(width=3.),
    )

    marker_flag_m: dict[str, str|float|dict[str,float]] = dict(
        size=20., symbol="circle-open", line=dict(width=3.), color="red", 
    ) 

    marker_sizes_c: list[float] = [10., 6., 4.]

    marker_sizes_h: list[float] = [3., 2., 2.5]

    marker_sizes_m: list[float] = [8., 10., 8., 11.]

    marker_sizes_map: list[float] = [10., 20.]

    marker_symbols_m: list[str] = ["circle", "circle", "diamond", "star"]

    months: list[str] = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    opacity_bias: float = 0.3

    opacity_hist: float = 0.7

    x_args: dict[str,str|float|int|bool] = dict(
        tickfont_size=14,
        showline=True,
        linewidth=1.,
        linecolor="black",
        mirror=True,
    )

    y_args: dict[str,str|float|int|bool] = dict(
        tickfont_size=14,
        title_standoff=5,
        showline=True,
        linewidth=1.,
        linecolor="black",
        mirror=True,
    )

    width_collapse: str = "70%"


def add_logo(
        fig: go.Figure, x: float, y: float, sx: float, sy: float
) -> go.Figure:
    """Add Empa logo to a figure
    :param fig: Plotly figure object
    :param x: Horizonal position as fraction of x axis
    :param y: Vertical position as fraction of y axis
    :param sx: Horizonal size as fraction of x axis
    :param sy: Vertical size as fraction of y axis
    :return: Plotly figure object
    """
    assets_path = AppConfig().assets_path.as_posix()
    fig.add_layout_image(
        dict(
            source=assets_path + "/logos/Logo_Empa.png",
            xref="paper",
            yref="paper",
            x=x,
            y=y,
            sizex=sx,
            sizey=sy,
            xanchor="left",
            yanchor="bottom",
            layer="above",
            opacity=0.3,
        )
    )

    return fig


def add_message_right(
        fig: go.Figure,
        msg: str,
        pos: tuple[float, float],
        color: str
) -> go.Figure:
    """Add right-aligned message to a plot
    :param fig: Plotly figure object
    :param msg: Message
    :param pos: (x,y) coordinates as fractions of plot dimensions
    :param color: Font color
    :return: Plotly figure object
    """
    fig.add_annotation(
        x=pos[0],
        y=pos[1],
        xref="paper",
        yref="paper",
        text=msg,
        showarrow=False,
        font=dict(size=10, color=color),
        xanchor="right",
    )

    return fig
