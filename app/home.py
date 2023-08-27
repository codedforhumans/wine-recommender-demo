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

def layout_home_sleep():
    return html.Div("Layout Home Sleeping")


def layout_home():
    get_new_data()
    return html.Div([
        "Hello Vyn World!",
        SPACE,
        "V3.0. Please give as many descriptors you would like for us to recommend you a wine.",
        SPACE,
        html.Div(id="inputs"),
        SPACE,
        html.Div(id="button-div"),
        SPACE,
        html.Div(id="pre-output"),
        SPACE,
        html.Div(id="result-div")]
                    ,style=style.HOME)


@app.callback(
    Output("result-div", "children"),
    Input("pre-output", "children"),
    Input("main_input", "value")
)
def result_div(pre_output, main_input):
    if pre_output is not None:
        layout_body = []
        if main_input is not None:
            result_lst = RECOMMEND_GLOBAL.run_recommender(main_input)
        for suggestion in result_lst:
            layout_body += [html.Div([
                html.Div(suggestion), 
                SPACE])]
        layout = html.Div(layout_body)
        return layout
    else:
        return None


@app.callback(
    Output("button-div", "children"),
    Input("content-div", "children")
)
def button_div(button_div):
    layout = html.Div(
        [dbc.Button("Get Recommendations", id="get-recommendation", size="sm", color="primary", className="me-1")]
    )
    return layout


@app.callback(
    Output("inputs", "children"),
    Input("content-div", "children")
)
def input_layout(content):
    layout = html.Div(
        [   SPACE_SMALL,
            # html.Div(id="geo-div-dev"),
            SPACE_SMALL,
            dcc.Input(id="main_input", type="text", placeholder="descriptors", style={'width':'90%'}, size='30'), #style={'marginRight':'30px'}
            SPACE_INPUTS
        ]
    )
    return layout

@app.callback(
    Output("geo-div-dev", "children"),
    Input("content-div", "children")
)
def get_geo_div(content):
    return html.Div("Returning Geo Div")

@app.callback(
    Output("pre-output", "children"),
    Input("main_input", "value"),
    # Input("variety-dropdown", "value"),
    # Input("country-dropdown", "value"),
    Input("get-recommendation", "n_clicks")
)
def update_output(main_input, n_click):
    print(n_click)
    if n_click is not None:
        if main_input == None:
            main_input_show = ""
        else:
            main_input_show = main_input

        return html.Div("Getting wine recommendations for the following: {}".format(main_input_show))
    else:
        return None





num_input = 3






# @app.callback(
#     Output("output", "children"),
#     Input("input1", "value"),
#     Input("input2", "value"),
#     Input("input3", "value"),
#     Input("input4", "value"),
#     Input("input5", "value"),
# )
# def update_output(input1, input2, input3, input4, input5):
#     return html.Div([html.Div("Hello"),
#                      f'Input 1 {input1}, Input 2 {input2}, Input 3 {input3}, Input 4 {input4}, and Input 5 {input5}'])
