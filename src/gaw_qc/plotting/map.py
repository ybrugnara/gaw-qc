import pandas as pd
import plotly.graph_objects as go
from gaw_qc.plotting.aesthetics import PlotSettings

psettings = PlotSettings()


def plot_map_of_stations(df: pd.DataFrame, cod: str | None) -> go.Figure:
    df["color"] = psettings.colors_map[0]
    df["size"] = psettings.marker_sizes_map[0]
    if cod is not None:
        df.loc[df.gaw_id == cod, "color"] = psettings.colors_map[1]
        df.loc[df.gaw_id == cod, "size"] = psettings.marker_sizes_map[1]
        df.sort_values(by="size", inplace=True)
    
    fig = go.Figure(
        go.Scattergeo(
            lon=df["longitude"],
            lat=df["latitude"],
            text=df["name"],
            customdata=df["variables"],
            hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
            marker=dict(
                size=df["size"],
                color=df["color"],
                opacity=1,
                line=dict(width=0),
            )
        )
    )

    fig.update_layout(
        width=520,
        height=270,
        margin=dict(l=0, r=0, t=0, b=0),
        geo=dict(
            scope="world",
            projection=go.layout.geo.Projection(type="natural earth"),
            showland=True,
            landcolor="rgb(217, 217, 217)",
            oceancolor="aliceblue",
            bgcolor="#f9f9f9",
        ),
    )

    return fig
