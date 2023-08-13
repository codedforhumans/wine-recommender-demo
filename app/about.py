from dash import Dash, dcc, html, Input, Output, ctx
import plotly.graph_objects as go
import plotly.io as pio
import dash_bootstrap_components as dbc
from flask import send_from_directory

pio.templates.default = "plotly_dark"

from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.io as pio

import pandas as pd

from config.template_functions import tabs_layout
import config.template_css as style

def layout_about():
    return html.Div("This is the About Section")