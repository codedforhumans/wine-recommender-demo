from dash import Dash
import dash_bootstrap_components as dbc

app = Dash(external_stylesheets=[dbc.themes.LUX], 
           meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        title = "Wine Recommender")