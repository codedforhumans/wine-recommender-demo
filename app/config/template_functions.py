import config.template_css as style
from dash import dcc
from dash import Dash, dcc, html, Input, Output, State, dash_table


def tabs_layout(tabs):
    tab_list = []
    for i in range(len(tabs)):
        tab_list.append(dcc.Tab(label=tabs[i], value=tabs[i], style=style.TAB, selected_style=style.TAB_SELECTED))
    if len(tabs) > 1:
        return dcc.Tabs(id='tabs', value=tabs[0], children=tab_list)
    else:
        return html.Div(dcc.Tabs(id='tabs', value=tabs[0], children=tab_list), style={"display": "none"})
    
    