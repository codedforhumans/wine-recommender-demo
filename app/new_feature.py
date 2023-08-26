from dash import Dash, dcc, html, Input, Output, ctx, State
import plotly.io as pio
import time
import dash_bootstrap_components as dbc
from flask import send_from_directory

pio.templates.default = "plotly_dark"

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from config.template_functions import tabs_layout
import config.template_css as style

from dash_init import app
from backend import recommender


####### NEW IMPORTS #################
import dash_leaflet as dl
#####################################

SPACE = html.Br()
SPACE_INPUTS = html.Div(style={"padding-bottom":"10px"})
SPACE_SMALL = html.Div(style={"padding":"5px"})

def get_new_data():
    global RECOMMEND_GLOBAL
    RECOMMEND_GLOBAL = recommender.Recommender()

UPDATE_INTERVAL = 3600

def get_new_data_interval():
    while True:
        time.sleep(UPDATE_INTERVAL)

executor = ThreadPoolExecutor(max_workers = 1)
executor.submit(get_new_data_interval)

def layout_new_sleep():
    return html.Div("Layout Home Sleeping")


def layout_new():
    get_new_data()
    return html.Div([
        "New Feature: Geography",
        html.Div(id="geo-div")])

@app.callback(
    Output("geo-div", "children"),
    Input("content-div", "children")
)
def get_geo_div(content):
    return html.Div([
    dl.Map([
        dl.TileLayer(), dl.LayerGroup(id="layer")],
           id="map", style={'width': '100%', 'height': '50vh', 
                            'margin': "auto", "display": "block"}),
])


@app.callback(
        Output("layer", "children"),
        [Input("map", "click_lat_lng")]
)
def map_click(click_lat_lng):
    return [dl.Marker(position=click_lat_lng, children=dl.Tooltip("({:.3f}, {:.3f})".format(*click_lat_lng)))]



# Read Data 
DATA_URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/geojson/vancouver-blocks.json"
 
# Generate polygon layer 
polygon = pydeck.Layer(
   "PolygonLayer",
   [[[-123.0, 49.196], [-123.0, 49.324], [-123.306, 49.324], [-123.306, 49.196]]],
   stroked=False,
   get_polygon="-",
   get_fill_color=[0, 0, 0, 20],
)
 
# Generate geojson from data set
geojson = pydeck.Layer(
   "GeoJsonLayer",
   DATA_URL,
   extruded=True,
   get_elevation="properties.valuePerSqm / 20",
   get_fill_color="[255, 255, properties.growth * 255]",
)
 
# Set initial state
INITIAL_VIEW_STATE = pydeck.ViewState(
   latitude=49.254, longitude=-123.13, zoom=11, max_zoom=16, pitch=45, bearing=0
)

# Combine the multiple layers
r = pydeck.Deck(
   layers=[polygon, geojson],
   initial_view_state=INITIAL_VIEW_STATE,
   mapbox_key=mapbox_api_token,
)
 
# Generate component to be passed into the Dash application
deck_component = (dgl(r.to_json(), id="deck-gl", mapboxKey=r.mapbox_key,))