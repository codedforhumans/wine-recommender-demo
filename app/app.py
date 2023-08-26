from dash import Dash, dcc, html, Input, Output, ctx
import plotly.graph_objects as go
import plotly.io as pio
import dash_bootstrap_components as dbc
from flask import send_from_directory

pio.templates.default = "plotly_dark"

# Check if the data directory exists, if not create it
import os
if not os.path.exists("app/data"):
    os.mkdir("app/data")

# Check if the data directory is empty, if so download the data file
# The data file is not included in the git repo because it is too large
# The data file is hosted on a private server
import requests
if len(os.listdir("app/data")) == 0:
    print("Downloading data...")
    url = "https://vyn.ai/recommender_data/data.tgz"
    user, password = "vyn_data", "myPassIsCool"
    r = requests.get(url, auth=(user, password))
    # unpack the data file with tar -xvzf data.tgz
    open("app/data/data.tgz", "wb").write(r.content)
    print("Download complete")

    # Unpack the data file
    import tarfile
    print("Unpacking data...")
    tar = tarfile.open("app/data/data.tgz")
    tar.extractall("app/data")
    tar.close()

    #remove the tar file
    os.remove("app/data/data.tgz")
    print("Unpacking complete")

from config.template_functions import tabs_layout
import config.template_css as style
from about import layout_about
from home import layout_home, layout_home_sleep
# from new_feature import layout_new

from dash_init import app

@app.callback(
    Output("content-div", "children"),
    Input("tabs", "value"),
)
def content(tab):
    if tab == "Home":
        return layout_home()
        # return layout_new()
        # return layout_home_sleep()
    elif tab == "About":
        return layout_about()

title = html.P("Vyn", style=style.TITLE)
# title = html.H1("Vyn")

# image_path = 'assets/logos/vyn_logo_large.jpeg'
# header = html.Div(html.Img(src=image_path, style=style.HEADER_LOGO), style=style.HEADER)

tabs = html.Div([tabs_layout(["Home", "About"])])

def layout():
    return html.Div([
        html.Div([
            html.Div([title], style=style.TOPBAR),
            html.Div([tabs], id="topbar-div", style=style.TOPBAR_MENU),  # Topbar (Tabs, Title, ...)
            html.Div("Loading Content...", id="content-div", style=style.CONTENT),  # Content (Loads in body() function)
            html.Div(id="data-div", style={"display": "none"}),  # Invisible Div to store data in json string format
            # html.Link(href='/assets/style.css', rel='stylesheet'),
        ], id="body", style=style.BODY),
    ], style={"width": "100vw", "height": "100vh", "align":"center", "justify-content":"center"})


app.layout = layout()
server = app.server

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=8050, debug=False)
