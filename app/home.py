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
# from ..backend import recommender

SPACE = html.Br()
SPACE_INPUTS = html.Div(style={"padding-bottom":"10px"})
SPACE_SMALL = html.Div(style={"padding":"5px"})

def get_new_data():
    global RECOMMEND_GLOBAL
    RECOMMEND_GLOBAL = recommender.Recommender()
    # COUNTRIES = RECOMMEND_GLOBAL.get_countries_min_count()
    # VARIETIES = RECOMMEND_GLOBAL.get_varieties_min_count()
    # any_variety = "Any Variety"
    # any_country = "Any Country"

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
        # SPACE,
        # html.Div(id="variety-div"),
        # SPACE,
        # html.Div(id="country-div"),
        SPACE,
        "Please give as many descriptors you would like for us to recommend you a wine.",
        SPACE,
        html.Div(id="inputs"),
        SPACE,
        html.Div(id="button-div"),
        SPACE,
        html.Div(id="pre-output"),
        SPACE,
        html.Div(id="result-div")]
                    ,style=style.HOME)


# @app.callback(
#     Output("variety-div", "children"),
#     Input("content-div", "children"),
#     # Input("country-dropdown", "value")
# )
# def variety_div(null): # country
#     result = [html.Div("Choose a Variety")]
#     variety = list(VARIETIES.keys())
#     # if country != any_country:
#     #     variety = VARIETIES[country]
#     dropdown_choices = [any_variety] + variety
#     dropdown = dcc.Dropdown(dropdown_choices, value = any_variety, id='variety-dropdown')
#     result += [dropdown]

#     return result


# @app.callback(
#     Output("country-div", "children"),
#     Input("content-div", "children"),
#     Input("variety-dropdown", "value")
# )
# def country_div(null, variety):
#     result = [html.Div("Choose a Country")]
#     country = list(COUNTRIES.keys())
#     if variety is not None:
#         print("variety is not None:")
#         if variety != any_variety:
#             print("variety != any_variety")
#             country = list(VARIETIES[variety])
#     dropdown_choices = [any_country] + country
#     dropdown = dcc.Dropdown(dropdown_choices, value = any_country, id='country-dropdown')
#     result += [dropdown]

#     return result


@app.callback(
    Output("result-div", "children"),
    Input("pre-output", "children"),
    Input("input_1", "value"),
    Input("input_2", "value"),
    Input("input_3", "value"),
    Input("input_4", "value"),
    Input("input_5", "value"),
    # Input("variety-dropdown", "value"),
    # Input("country-dropdown", "value")
)
def result_div(pre_output, input_1, input_2, input_3, input_4, input_5):
    if pre_output is not None:
        layout_body = []
        all_inputs = []
        for input in [input_1, input_2, input_3, input_4, input_5]:
            if input is not None:
                all_inputs += [input]
        result_lst = RECOMMEND_GLOBAL.run_recommender(all_inputs)
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
            # SPACE_SMALL,
            dcc.Input(id="input_1", type="text", placeholder="descriptor 1", style={'marginRight':'10px'}),
            SPACE_INPUTS,
            dcc.Input(id="input_2", type="text", placeholder="descriptor 2", style={'marginRight':'10px'}),
            SPACE_INPUTS,
            dcc.Input(id="input_3", type="text", placeholder="descriptor 3", style={'marginRight':'10px'}),
            SPACE_INPUTS,
            dcc.Input(id="input_4", type="text", placeholder="descriptor 4", style={'marginRight':'10px'}),
            SPACE_INPUTS,
            dcc.Input(id="input_5", type="text", placeholder="descriptor 5", style={'marginRight':'10px'}),
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
    Input("input_1", "value"),
    Input("input_2", "value"),
    Input("input_3", "value"),
    Input("input_4", "value"),
    Input("input_5", "value"),
    # Input("variety-dropdown", "value"),
    # Input("country-dropdown", "value"),
    Input("get-recommendation", "n_clicks")
)
def update_output(input_1, input_2, input_3, input_4, input_5, n_click):
    print(n_click)
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
        if input_4 == None:
            input_4_show = ""
        else:
            input_4_show = input_4
        if input_5 == None:
            input_5_show = ""
        else:
            input_5_show = input_5

        return html.Div("Getting wine recommendations for the following: {} {} {} {} {}".format(input_1_show, input_2_show, input_3_show, input_4_show, input_5_show))
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
