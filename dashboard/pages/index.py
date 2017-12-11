import pandas as pd
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import dash_table_experiments as dt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def render_index_layout():
    layout = html.Div([
        html.Div([
            html.H1('Research',
                style={'display': 'inline',
                       'float': 'left',
                       'font-size': '2.65em',
                       'margin-left': '7px',
                       'font-weight': 'bolder',
                       'font-family': 'Open Sans Light',
                       'color': "rgba(117, 117, 117, 0.95)",
                       'margin-top': '20px',
                       'margin-bottom': '40px'
                       })
        ], className='row', style={'margin-bottom': '2%'}),
        html.H4(html.A('Noise analysis', href="/noise_analysis")),
    ])
    return layout