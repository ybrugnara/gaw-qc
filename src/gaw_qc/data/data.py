import dash
from dash import Dash, dash_table, html, dcc, callback, Input, Output

dash.register_page(__name__)

app = Dash(__name__)
app.layout = dash_table.DataTable(
    df.to_dict("records"), [{"name": i, "id": i} for i in df.columns]
)
