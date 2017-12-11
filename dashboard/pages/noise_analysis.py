import pandas as pd
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc


def render_noise_analysis_layout():
    header = html.Div(
        className='header',
        children=[
            html.A([html.Img(src='/static/back_arrow.png', style={'height': '40px', 'margin-left': '5px'}),
                    # html.Div('',
                    #          style={'font-size': '16px', 'font-weight': '500',
                    #                 'font-family': 'Open Sans', 'color': "#fff", 'letter-spacing': '0.5px',
                    #                 'text-transform': 'uppercase'})
                    ],
                   style={'display': 'flex', 'align-items': 'center', 'text-decoration': 'none'},
                   id='back_to_companies_link', className="link", href="/"),
            html.A(
                html.H3('Noise analysis',
                        style={'display': 'inline',
                               'float': 'left',
                               'font-size': '24',
                               'font-weight': 'bolder',
                               'font-family': 'Open Sans Light',
                               'color': "rgba(117, 117, 117, 0.95)",
                               'margin-top': '20px',
                               'margin-bottom': '30px'
                               }),
                href='/noise_analysis',
                style={'text-decoration': 'none'}
            ),
            html.Div(
                className="links",
                children=[
                    html.Ul([
                        html.A('             ', className="link active", href="/page1"),
                        html.A('             ', className="link", href="/page2"),
                        html.A('              ', className="link", href="/page3"),
                    ])
                ]
            )
        ]
    )

    layout = html.Div([
        header,
    ])
    return layout


