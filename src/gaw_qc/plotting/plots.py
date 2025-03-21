import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
from typing import Literal
from gaw_qc.data.classes import PlottingData
from gaw_qc.db.variables import GawVars, GawUnits
from gaw_qc.models.model_config import ModelSettings
from gaw_qc.plotting.aesthetics import add_message, PlotSettings
from gaw_qc.plotting.base_plots import empty_subplot

msettings = ModelSettings()
psettings = PlotSettings()


def plot_hourly(
        fig: go.Figure,
        df: pd.DataFrame,
        param: GawVars,
        use_cams: bool,
        points_on: bool,
        extra: bool,
        row: int,
        col: int,
) -> go.Figure:
    """Make plot of hourly time series
    :param fig: Figure object
    :param df: Data frame with measurements, CAMS data, and flags
    :param param: Variable code
    :param use_cams: Whether to plot CAMS data
    :param points_on: Whether to plot points instead of lines
    :param extra: Whether an extra series was uploaded by the user
    :param row: row index for subplots
    :param col: col index for subplots
    :return: Figure object
    """
    # Plot measurements
    if points_on:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[param],
                mode="markers",
                marker_color=psettings.colors_h[0],
                marker_size=psettings.marker_sizes_h[0],
                hoverinfo="skip",
                name=psettings.labels_plot[0],
            ),
            row=row,
            col=col,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[param],
                mode="lines",
                line_color=psettings.colors_h[0],
                line_width=psettings.line_widths_h[0],
                hoverinfo="skip",
                name=psettings.labels_plot[0],
            ),
            row=row,
            col=col,
        )

    # Plot additionally uploaded series
    if extra:
        if points_on:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df.iloc[:, 1],
                    mode="markers",
                    marker_color=psettings.colors_h[3],
                    marker_size=psettings.marker_sizes_h[3],
                    hoverinfo="skip",
                    name=df.columns[1],
                ),
                row=row,
                col=col,
                secondary_y=True,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df.iloc[:, 1],
                    mode="lines",
                    line_color=psettings.colors_h[3],
                    line_width=psettings.line_widths_h[3],
                    hoverinfo="skip",
                    name=df.columns[1],
                ),
                row=row,
                col=col,
                secondary_y=True,
            )
    
    # Plot CAMS
    if use_cams:
        if points_on:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[param + "_cams"],
                    mode="markers",
                    hoverinfo="skip",
                    marker_color=psettings.colors_h[1],
                    marker_size=psettings.marker_sizes_h[1],
                    name=psettings.labels_plot[1],
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["CAMS+"],
                    mode="markers",
                    hoverinfo="skip",
                    marker_color=psettings.colors_h[2],
                    marker_size=psettings.marker_sizes_h[2],
                    name=psettings.labels_plot[2],
                ),
                row=row,
                col=col,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[param + "_cams"],
                    mode="lines",
                    hoverinfo="skip",
                    line_color=psettings.colors_h[1],
                    line_width=psettings.line_widths_h[1],
                    name=psettings.labels_plot[1],
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["CAMS+"],
                    mode="lines",
                    hoverinfo="skip",
                    line_color=psettings.colors_h[2],
                    line_width=psettings.line_widths_h[2],
                    name=psettings.labels_plot[2],
                ),
                row=row,
                col=col,
            )
    
        # Plot shaded areas for significant differences
        if "Flag CAMS" in df.columns:
            flags_cum_yellow = df.index[df["Flag CAMS"]==1]
            if len(flags_cum_yellow) > 0:
                flags_cum = (flags_cum_yellow - df.index[0]).astype(
                    "int64"
                ) / 3.6e12  # convert to number of hours from time 0
                flags_cum_diff = np.diff(
                    flags_cum, prepend=-msettings.window_size_cams
                )
                i_blocks = np.array(
                    np.where(flags_cum_diff >= msettings.window_size_cams),
                    ndmin=1
                )
                if i_blocks.size == 1:
                    flags_cum = np.array(flags_cum[i_blocks.item()], ndmin=1)
                else:
                    i_blocks = i_blocks.squeeze()
                    flags_cum = flags_cum[i_blocks]
                last_pos = 0
                for i, fl in enumerate(flags_cum):
                    t_flag = df.index[0] + timedelta(hours=fl)
                    if i == i_blocks.size - 1:
                        n = np.sum(flags_cum_diff[last_pos + 1 :])
                    else:
                        n = np.sum(
                            flags_cum_diff[i_blocks[i] + 1 : i_blocks[i + 1]]
                        )
                        last_pos = i_blocks[i + 1]
                    t1 = t_flag + timedelta(hours=n)
                    fig.add_vrect(
                        x0=t_flag,
                        x1=t1,
                        fillcolor=psettings.color_bias,
                        opacity=psettings.opacity_bias,
                        line_width=0,
                        row=row,
                        col=col,
                    )

        # Add message if CAMS data are missing
        if df[param + "_cams"].count() == 0:
            fig = add_message(
                fig,
                msg="(CAMS data not available for the analyzed period)",
                pos=(0, 1.17),
                color=psettings.colors_h[1],
                align="left"
            )
        if df["CAMS+"].count() == 0:
            fig = add_message(
                fig,
                msg="(CAMS+ data not available for the analyzed period)",
                pos=(0, 1.13),
                color=psettings.colors_h[2],
                align="left"
            )

    
    # Plot flags
    fig.add_trace(
        go.Scatter(
            x=df.index[df["Flag LOF"]==1],
            y=df.loc[df["Flag LOF"]==1, param],
            mode="markers",
            marker=psettings.marker_flag_h,
            marker_color=psettings.colors_pie[1],
            name=psettings.labels_plot[3],
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index[df["Flag all"]>1],
            y=df.loc[df["Flag all"]>1, param],
            mode="markers",
            marker=psettings.marker_flag_h,
            marker_color=psettings.colors_pie[2],
            name=psettings.labels_plot[4],
        ),
        row=row,
        col=col,
    )

    # Format axes
    title = param.upper() + " mole fraction (" + GawUnits[param].value + ")"
    fig.update_xaxes(
        title_text="Time UTC",
        row=row,
        col=col,
        **psettings.x_args
    )
    fig.update_yaxes(
        title_text=title,
        row=row,
        col=col,
        **psettings.y_args
    )
    if extra:
        fig.update_yaxes(
            title_text=df.columns[1],
            title_font_color=psettings.colors_h[3],
            tickfont_color=psettings.colors_h[3],
            showgrid=False,
            row=row,
            col=col,
            secondary_y=True
        )

    return fig


def plot_pie(
        fig: go.Figure,
        df: pd.DataFrame,
        param: GawVars,
        row: int,
        col: int,
) -> go.Figure:
    """Make pie chart
    :param fig: Figure object
    :param df: Data frame with measurements, CAMS data, and flags
    :param param: Variable code
    :param row: row index for subplots
    :param col: col index for subplots
    :return: Figure object
    """
    # Put data to plot in a list
    data_pie = [
        df.loc[df["Flag all"]==0, param].count(),
        (df["Flag all"] == 1).sum(),
        (df["Flag all"] == 2).sum(),
    ]

    # Plot pie
    fig.add_trace(
        go.Pie(
            labels=psettings.labels_pie,
            values=data_pie,
            marker={"colors": psettings.colors_pie},
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.update_traces(name="", row=row, col=col)

    return fig


def plot_histogram(
        fig: go.Figure,
        df: pd.DataFrame,
        param: GawVars,
        bin_size: float,
        use_cams: bool,
        row: int,
        col: int,
) -> go.Figure:
    """Make histogram
    :param fig: Figure object
    :param df: Data frame with measurements, CAMS data, and flags
    :param param: Variable code
    :param bin_size: Size of bins
    :param use_cams: Whether to plot CAMS data
    :param row: row index for subplots
    :param col: col index for subplots
    :return: Figure object
    """
    # Plot measurements
    fig.add_trace(
        go.Histogram(
            x=df[param],
            histnorm="probability",
            name=psettings.labels_plot[0],
            marker_color=psettings.colors_h[0],
            xbins=dict(size=bin_size),
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    
    # Plot CAMS
    if use_cams:
        fig.add_trace(
            go.Histogram(
                x=df[param + "_cams"],
                histnorm="probability",
                name=psettings.labels_plot[1],
                marker_color=psettings.colors_h[1],
                xbins=dict(size=bin_size),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Histogram(
                x=df["CAMS+"],
                histnorm="probability",
                name=psettings.labels_plot[2],
                marker_color=psettings.colors_h[2],
                xbins=dict(size=bin_size),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    # Format axes and opacity
    title = param.upper() + " mole fraction (" + GawUnits[param].value + ")"
    fig.update_xaxes(
        title_text=title,
        showgrid=True,
        row=row,
        col=col,
        **psettings.x_args
    )
    fig.update_yaxes(
        title_text="Density",
        row=row,
        col=col,
        **psettings.y_args
    )
    fig.update_traces(opacity=psettings.opacity_hist, row=row, col=col)

    return fig


def plot_monthly(
        fig: go.Figure,
        df: pd.DataFrame,
        y_pred_mon: pd.Series,
        param: GawVars,
        mtp: int,
        use_cams: bool,
        points_on: bool,
) -> go.Figure:
    """Make plot of monthly time series
    :param fig: Figure object
    :param df: Data frame with measurements, SARIMA predictions, CAMS data, and flags
    :param y_pred_mon: CAMS+ predictions
    :param param: Variable code
    :param mtp: Number of months to predict
    :param use_cams: Whether to plot CAMS data
    :param points_on: Whether to plot points instead of lines
    :return: Figure object
    """
    df_test = df.iloc[-mtp:,:]

    # Plot SARIMA
    if (mtp > 1) & (not points_on):  # plot a line + filled area
        fig.add_trace(
            go.Scatter(
                x=df_test.index,
                y=df_test["lower"],
                mode="lines",
                line_color=psettings.colors_m[1],
                line_width=0.1,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_test.index,
                y=df_test["upper"],
                mode="lines",
                line_color=psettings.colors_m[1],
                line_width=0.1,
                fill="tonextx",
                hoverinfo="skip",
                name=str(int(100 * (1 - msettings.p_conf)))
                + "% confidence range",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_test.index,
                y=df_test["prediction"],
                mode="lines",
                line_color=psettings.colors_m[1],
                line_width=psettings.line_widths_m[1],
                name=psettings.labels_plot[5],
            )
        )
    else: # plot a point + error bars
        df_test["upper"] -= df_test["prediction"]
        fig.add_trace(
            go.Scatter(
                x=df_test.index,
                y=df_test["prediction"],
                mode="markers",
                marker_color=psettings.colors_m[1],
                marker_size=psettings.marker_sizes_m[1],
                marker_symbol=psettings.marker_symbols_m[1],
                error_y=dict(
                    type="data",
                    visible=True,
                    array=df_test["upper"],
                ),
                name=psettings.labels_plot[5],
            )
        )

    # Plot CAMS
    if use_cams:
        fig.add_trace(
            go.Scatter(
                x=y_pred_mon.index,
                y=y_pred_mon,
                mode="markers",
                marker_color=psettings.colors_m[3],
                marker_size=psettings.marker_sizes_m[3],
                marker_symbol=psettings.marker_symbols_m[3],
                name=psettings.labels_plot[2],
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_test.index,
                y=df_test[param + "_cams"],
                mode="markers",
                marker_color="white",
                marker_size=psettings.marker_sizes_m[2],
                marker_symbol=psettings.marker_symbols_m[2],
                marker_line_color=psettings.colors_m[2],
                marker_line_width=2,
                name=psettings.labels_plot[1],
            )
        )

        # Add message if CAMS data are missing
        if df_test[param + "_cams"].count() == 0:
            fig = add_message(
                fig,
                msg="(CAMS data not available for the analyzed period)",
                pos=(-0.1, 1.17),
                color=psettings.colors_h[1],
                align="left"
            )
        # if y_pred_mon.count() == 0:
        #     fig = add_message(
        #         fig,
        #         msg="(CAMS+ data not available for the analyzed period)",
        #         pos=(-0.1, 1.13),
        #         color=psettings.colors_h[2],
        #         align="left"
        #     )

    # Plot measurements
    if points_on:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[param],
                mode="markers",
                marker_color=psettings.colors_m[0],
                marker_size=psettings.marker_sizes_m[0],
                marker_symbol=psettings.marker_symbols_m[0],
                name=psettings.labels_plot[0],
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[param],
                mode="lines+markers",
                line_color=psettings.colors_m[0],
                line_width=psettings.line_widths_m[0],
                marker_color=psettings.colors_m[0],
                marker_size=psettings.marker_sizes_m[0],
                marker_symbol=psettings.marker_symbols_m[0],
                name=psettings.labels_plot[0],
            )
        )

    # Plot flags
    if df_test["flag"].sum() > 0:
        fig.add_trace(
            go.Scatter(
                x=df_test.index[df_test["flag"]],
                y=df_test.loc[df_test["flag"], param],
                mode="markers",
                marker=psettings.marker_flag_m,
                name="Flag",
                showlegend=False,
            )
        )

    # Format axes
    fig.update_xaxes(**psettings.x_args)
    fig.update_yaxes(**psettings.y_args)

    return fig


def plot_diurnal(
        fig: go.Figure,
        plot_data: PlottingData,
        res: Literal["hourly", "monthly"],
        tz: str,
        points_on: bool,
        row: int,
        col: int,
) -> go.Figure:
    """Make plot of diurnal cycle
    :param fig: Figure object
    :param plot_data: Data to be plotted
    :param res: Temporal resolution of the data
    :param tz: Timezone
    :param points_on: Whether to plot points instead of lines
    :param row: row index for subplots
    :param col: col index for subplots
    :return: Figure object
    """
    colors = psettings.colors_years

    if res == "hourly":
        x_labels = pd.date_range("2020-01-01", periods=24, freq="h")
        i_col = len(colors) - len(plot_data.years_for_mean)
        for iy in plot_data.years_for_mean:
            if points_on:
                fig.add_trace(
                    go.Scatter(
                        x=x_labels,
                        y=plot_data.diurnal_cycle[str(iy)],
                        mode="markers",
                        marker_size=psettings.marker_sizes_c[2],
                        marker_color=colors[i_col],
                        customdata=plot_data.diurnal_cycle[str(iy) + "_n"],
                        hovertemplate="%{y} (N=%{customdata})",
                        name=str(iy),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x_labels,
                        y=plot_data.diurnal_cycle[str(iy)],
                        mode="lines",
                        line_width=psettings.line_widths_c[2],
                        line_color=colors[i_col],
                        customdata=plot_data.diurnal_cycle[str(iy) + "_n"],
                        hovertemplate="%{y} (N=%{customdata})",
                        name=str(iy),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
            i_col += 1
        if len(plot_data.years_for_mean) > 1:
            if points_on:
                fig.add_trace(
                    go.Scatter(
                        x=x_labels,
                        y=plot_data.diurnal_cycle["multiyear"],
                        mode="markers",
                        marker_size=psettings.marker_sizes_c[1],
                        marker_color=psettings.colors_c[1],
                        name=plot_data.period_label,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=x_labels,
                        y=plot_data.diurnal_cycle["multiyear"],
                        mode="lines",
                        line_width=psettings.line_widths_c[1],
                        line_color=psettings.colors_c[1],
                        name=plot_data.period_label,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=plot_data.diurnal_cycle["test"],
                mode="markers",
                marker_color=psettings.colors_c[0],
                marker_size=psettings.marker_sizes_c[0],
                marker_symbol="circle",
                customdata=plot_data.diurnal_cycle["test_n"],
                hovertemplate="%{y} (N=%{customdata})",
                name=plot_data.label,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(
            title_text="Time " + tz,
            tickformat="%H:%M",
            row=row,
            col=col,
            **psettings.x_args
        )
        fig.update_yaxes(
            title_text=plot_data.y_title,
            row=row,
            col=col,
            **psettings.y_args
        )

    else:
        fig = empty_subplot(fig, "No hourly data", row, col)

    return fig


def plot_seasonal(
        fig: go.Figure,
        plot_data: PlottingData,
        points_on: bool,
        row: int,
        col: int,
) -> go.Figure:
    """Make plot of seasonal cycle
    :param fig: Figure object
    :param plot_data: Data to be plotted
    :param points_on: Whether to plot points instead of lines
    :param row: row index for subplots
    :param col: col index for subplots
    :return: Figure object
    """
    colors = psettings.colors_years

    i_col = len(colors) - len(plot_data.years_for_mean)
    for iy in plot_data.years_for_mean:
        if points_on:
            fig.add_trace(
                go.Scatter(
                    x=plot_data.seasonal_cycle.index,
                    y=plot_data.seasonal_cycle[str(iy)],
                    mode="markers",
                    marker_size=psettings.marker_sizes_c[2],
                    marker_color=colors[i_col],
                    name=str(iy),
                ),
                row=row,
                col=col,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=plot_data.seasonal_cycle.index,
                    y=plot_data.seasonal_cycle[str(iy)],
                    mode="lines",
                    line_width=psettings.line_widths_c[2],
                    line_color=colors[i_col],
                    name=str(iy),
                ),
                row=row,
                col=col,
            )
        i_col += 1
    if len(plot_data.years_for_mean) > 1:
        if points_on:
            fig.add_trace(
                go.Scatter(
                    x=plot_data.seasonal_cycle.index,
                    y=plot_data.seasonal_cycle["multiyear"],
                    mode="markers",
                    marker_size=psettings.marker_sizes_c[1],
                    marker_color=psettings.colors_c[1],
                    name=plot_data.period_label,
                ),
                row=row,
                col=col,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=plot_data.seasonal_cycle.index,
                    y=plot_data.seasonal_cycle["multiyear"],
                    mode="lines",
                    line_width=psettings.line_widths_c[1],
                    line_color=psettings.colors_c[1],
                    name=plot_data.period_label,
                ),
                row=row,
                col=col,
            )
    fig.add_trace(
        go.Scatter(
            x=plot_data.seasonal_cycle.index,
            y=plot_data.seasonal_cycle["test"],
            mode="markers",
            marker_color=psettings.colors_c[0],
            marker_size=psettings.marker_sizes_c[0],
            marker_symbol="circle",
            name=plot_data.label,
        ),
        row=row,
        col=col,
    )

    fig.update_xaxes(
        title_text="Month",
        ticktext=psettings.months,
        tickvals=np.arange(1, 13),
        range=[0, 13],
        showgrid=False,
        row=row,
        col=col,
        **psettings.x_args
    )
    fig.update_yaxes(
        title_text=plot_data.y_title,
        row=row,
        col=col,
        **psettings.y_args
    )

    return fig


def plot_variability(
        fig: go.Figure,
        plot_data: PlottingData,
        res: Literal["hourly", "monthly"],
        points_on: bool,
        row: int,
        col: int,
) -> go.Figure:
    """Make plot of variability cycle
    :param fig: Figure object
    :param plot_data: Data to be plotted
    :param res: Temporal resolution of the data
    :param points_on: Whether to plot points instead of lines
    :param row: row index for subplots
    :param col: col index for subplots
    :return: Figure object
    """
    colors = psettings.colors_years

    if res == "hourly":
        i_col = len(colors) - len(plot_data.years_for_mean)
        for iy in plot_data.years_for_mean:
            if points_on:
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.var_cycle.index,
                        y=plot_data.var_cycle[str(iy)],
                        mode="markers",
                        marker_size=psettings.marker_sizes_c[2],
                        marker_color=colors[i_col],
                        name=str(iy),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.var_cycle.index,
                        y=plot_data.var_cycle[str(iy)],
                        mode="lines",
                        line_width=psettings.line_widths_c[2],
                        line_color=colors[i_col],
                        name=str(iy),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
            i_col += 1
        if len(plot_data.years_for_mean) > 1:
            if points_on:
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.var_cycle.index,
                        y=plot_data.var_cycle["multiyear"],
                        mode="markers",
                        marker_size=psettings.marker_sizes_c[1],
                        marker_color=psettings.colors_c[1],
                        name=plot_data.period_label,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=plot_data.var_cycle.index,
                        y=plot_data.var_cycle["multiyear"],
                        mode="lines",
                        line_width=psettings.line_widths_c[1],
                        line_color=psettings.colors_c[1],
                        name=plot_data.period_label,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
        fig.add_trace(
            go.Scatter(
                x=plot_data.var_cycle.index,
                y=plot_data.var_cycle["test"],
                mode="markers",
                marker_color=psettings.colors_c[0],
                marker_size=psettings.marker_sizes_c[0],
                marker_symbol="circle",
                name=plot_data.label,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(
            title_text="Month",
            ticktext=psettings.months,
            tickvals=np.arange(1, 13),
            range=[0, 13],
            showgrid=False,
            row=row,
            col=col,
            **psettings.x_args
        )
        fig.update_yaxes(
            title_text="Standard deviation of hourly " + plot_data.y_title,
            row=row,
            col=col,
            **psettings.y_args
        )

    else:
        fig = empty_subplot(fig, "No hourly data", row, col)

    return fig
    