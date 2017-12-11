import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from dashboard.app import app

np.random.seed(42)

# Neural Network
model = Sequential()

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
        dcc.Graph(id='graph-with-slider', className='row'),
        html.Div([
            html.Div([
                html.H4('Noise std:'),
                dcc.Slider(
                    id='target-noise-slider',
                    min=0,
                    max=1,
                    value=0,
                    step=0.05
                ),
            ], className='four columns'),
            html.Div([
                html.H4('X range:'),
                dcc.RangeSlider(
                    id='sin-scale-slider',
                    min=np.pi,
                    max=10 * np.pi,
                    step=1 / 2 * np.pi,
                    marks={
                        0: {'label': '0'},
                        np.pi: {'label': 'Pi'},
                        1 * np.pi: {'label': '1Pi'},
                        2 * np.pi: {'label': '2Pi'},
                        3 * np.pi: {'label': '3Pi'},
                        4 * np.pi: {'label': '4Pi'},
                        5 * np.pi: {'label': '5Pi'},
                        6 * np.pi: {'label': '6Pi'},
                        7 * np.pi: {'label': '7Pi'},
                        8 * np.pi: {'label': '8Pi'},
                        9 * np.pi: {'label': '9Pi'},
                        10 * np.pi: {'label': '10Pi'}
                    },
                    value=[0, 5 * np.pi]
                ),
            ], className='four columns'),
        ], className='row', style={'margin-bottom': '50px', 'margin-left': '10px'}),
        html.Div([
            dcc.Input(id='input-1', type='text', value=30, className='four columns'),
            dcc.Input(id='input-2', type='text', value=30, className='four columns'),
        ], className='row', style={'margin-bottom': '50px', 'margin-left': '10px'}),
        html.Div([
            html.Button(id='submit-button', n_clicks=0, children='Start NN'),
        ], className='row', style={'margin-bottom': '50px', 'margin-left': '10px'})
    ])

    return layout


@app.callback(
    dash.dependencies.Output('graph-with-slider', 'figure'),
    [dash.dependencies.Input('target-noise-slider', 'value'),
     dash.dependencies.Input('sin-scale-slider', 'value'),
     dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-1', 'value'),
     dash.dependencies.State('input-2', 'value')]
)
def update_figure(selected_noise, sin_scale_value, n_clicks, input1, input2):

    np.random.seed(42)

    # Creating Input & Output
    x = np.linspace(sin_scale_value[0], sin_scale_value[1], num=1000)
    target = np.sin(x)

    # Introducing the noise in output
    sigma_tg = selected_noise
    mu_tg=0
    noise_tg = np.random.normal(mu_tg, sigma_tg, len(x))
    tg_noise = target + noise_tg

    traces = []

    if n_clicks:

        x_train, x_test, tg_train, tg_test = train_test_split(
            x, tg_noise, test_size=0.3, random_state=42)

        # Introducing noise in training input
        mu_x, sigma_x = 0, 0.1
        noise_x = np.random.normal(mu_x, sigma_x, len(x_train))
        x_noise = x_train + noise_x

        model.add(Dense(int(input1), activation='tanh', input_shape=(1,)))
        model.add(Dense(int(input2), activation='tanh'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mse', 'mape'])
        history = model.fit(x_noise, tg_train,
                            batch_size=100,
                            epochs=100,
                            verbose=0)
        # Comparing results
        NN_pred_test = model.predict(x_test).reshape(1, -1)
        NN_pred_train = model.predict(x_train).reshape(1, -1)

        trace_test = go.Scatter(
            x=x_test,
            y=NN_pred_test[0],
            mode='markers',
            name='NN_test',
            marker=dict(size=10,
                        line=dict(width=3),
                        opacity=0.5
                        ))

        trace_train = go.Scatter(
            x=x_train,
            y=NN_pred_train[0],
            mode='markers',
            name='NN_train',
            marker=dict(size=5,
                        line=dict(width=1),
                        opacity=0.5
                        ))
        traces.extend([trace_train, trace_test])

    trace_target = go.Scatter(
        x=x,
        y=target,
        mode='markers',
        name='target')

    trace_noise = go.Scatter(
        x=x,
        y=tg_noise,
        mode='markers',
        name='target+noise')

    traces.extend([trace_target, trace_noise])

    return {'data': traces}

