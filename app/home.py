from dash import Dash, dcc, html, Input, Output, ctx
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
# from ..backend import recommender

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

def layout_home():
    get_new_data()
    return html.Div([
        "Hello Vyn World!",
        SPACE,
        "Please give as many descriptors you would like for us to recommend you a wine.",
        SPACE,
        html.Div(id="inputs"),
        SPACE,
        html.Div(id="button-div"),
        SPACE,
        html.Div(id="output"),
        SPACE,
        html.Div(id="result-div")]
                    ,style=style.HOME)

@app.callback(
    Output("result-div", "children"),
    Input("output", "children"),
    Input("input_1", "value"),
    Input("input_2", "value"),
    Input("input_3", "value")
)
def result_div(output, input_1, input_2, input_3):
    if output is not None:
        layout_body = []
        result_lst = RECOMMEND_GLOBAL.run_recommender([input_1, input_2, input_3])
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
    layout = html.Div(dbc.Button("Get Recommendations", id="get-recommendation", size="sm", color="primary", className="me-1"),
    )
    return layout


@app.callback(
    Output("inputs", "children"),
    Input("content-div", "children")
)
def input_layout(content):
    layout = html.Div(
        [
            SPACE_SMALL,
            dcc.Input(id="input_1", type="text", placeholder="descriptor 1", style={'marginRight':'10px'}),
            SPACE_INPUTS,
            dcc.Input(id="input_2", type="text", placeholder="descriptor 2", style={'marginRight':'10px'}),
            SPACE_INPUTS,
            dcc.Input(id="input_3", type="text", placeholder="descriptor 3", style={'marginRight':'10px'})
        ]
    )
    return layout

@app.callback(
    Output("output", "children"),
    Input("input_1", "value"),
    Input("input_2", "value"),
    Input("input_3", "value"),
    Input("get-recommendation", "n_clicks")
)
def update_output(input_1, input_2, input_3, n_click):
    if n_click is not None:
        if input_1 == None:
            input_1_show = ""
        else:
            input_1_show = input_1
        if input_2 == None:
            input_2_show = ""
        else:
            input_2_show = input_2
        if input_3 == None:
            input_3_show = ""
        else:
            input_3_show = input_3

        return html.Div("Getting wine recommendations for the following: {} {} {}".format(input_1_show, input_2_show, input_3_show))
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
