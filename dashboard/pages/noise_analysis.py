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

# Creating Input & Output
np.random.seed(42)
x = np.linspace(0, 3*np.pi, num=1000)
target = np.sin(x)

# Neural Network
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
        html.Div([
            html.Div([
                dcc.Graph(id='graph')], className='six columns'),
            html.Div([
                dcc.Graph(id='graph2')], className='six columns')
            ], className='row'),
        html.Div([
                html.Div(id='output-a' , className='four columns'),
                html.Div(id='output-b', className='four columns')
        ], className='row', style={'margin-bottom': '10px', 'margin-left': '10px'}),
        html.Div([
            html.Div([
                dcc.Slider(
                    id='target-noise-slider',
                    min=0,
                    max=4,
                    value=0,
                    step=0.1,
                    marks={
                        0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                    },
                ),
            ], className='four columns'),
            html.Div([
                dcc.Slider(
                    id='x-noise-slider',
                    min=0,
                    max=4,
                    value=0,
                    step=0.1,
                    marks={
                        0: {'label': '0'},
                        1: {'label': '1'},
                        2: {'label': '2'},
                        3: {'label': '3'},
                        4: {'label': '4'},
                    },

                ),
            ], className='four columns'),
        ], className='row', style={'margin-bottom': '50px', 'margin-left': '10px'}),
        html.Div([
            html.Div(id='output-container-button_1',
                     children='Neurons in first layer', className='four columns'),
            html.Div(id='output-container-button_2',
                     children='Neurons in second layer', className='four columns'),
            html.Div(id='output-container-button_3',
                     children='Number of epochs', className='four columns')
        ], className='row', style={'margin-bottom': '2px','margin-left': '10px'}),
        html.Div([
            dcc.Input(id='input-1', type='text', value=100, className='four columns'),
            dcc.Input(id='input-2', type='text', value=100, className='four columns'),
            dcc.Input(id='input-3', type='text', value=200, className='four columns')
        ], className='row', style={'margin-bottom': '50px', 'margin-left': '10px'}),
        html.Div([
            dcc.Dropdown(id='dropdown-activation',
                         options=[
                             {'label': 'relu', 'value': 'relu'},
                             {'label': 'tanh', 'value': 'tanh'}
                         ],
                         value='tanh', className='four columns'
                         )
        ], className='row', style={'margin-bottom': '30px', 'margin-left': '1330px'}),
        html.Div([
            html.Button(id='submit-button', n_clicks=0, children='Start NN', className='four columns'),
            ], className='row', style={'margin-bottom': '20px', 'margin-left': '700px'}),
        html.Div([
            dcc.Graph(id='loss-function')], className='row')
    ])

    return layout

@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    [dash.dependencies.Input('target-noise-slider', 'value'),
     ]
)
def update_figure(tg_sd_noise):

    # Introducing noise in output
    noise_tg = np.random.normal(0, tg_sd_noise, len(x))
    tg_noise = target + noise_tg

    traces = []

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

@app.callback(
    dash.dependencies.Output('output-a', 'children'),
    [dash.dependencies.Input('target-noise-slider', 'value')])
def callback_a(value):
    return 'Noise in output (SD): "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('output-b', 'children'),
    [dash.dependencies.Input('x-noise-slider', 'value')])
def callback_a(value):
    return 'Noise in training input (SD): "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('graph2', 'figure'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('graph', 'figure'),
    dash.dependencies.State('target-noise-slider', 'value'),
    dash.dependencies.State('x-noise-slider', 'value'),
    dash.dependencies.State('input-1', 'value'),
    dash.dependencies.State('input-2', 'value'),
    dash.dependencies.State('input-3', 'value'),
    dash.dependencies.State('dropdown-activation', 'value')
     ]
)

def update_NN(n_clicks, graph, tg_sd_noise,x_sd_noise, input1, input2, input3, dropdown):
    traces2 = []
    test_mse=''
    train_mse=''
    train_mape=''
    test_mape=''
    if graph is not None:
        traces2.extend(graph['data'])
    if n_clicks:
        np.random.seed(42)

        # Introducing noise in output
        noise_tg = np.random.normal(0, tg_sd_noise, len(x))
        tg_noise = target + noise_tg

        # Splitting data
        x_train, x_test, tg_train, tg_test = train_test_split(
            x, tg_noise, test_size=0.3, random_state=42)

        # Putting noise into training input
        noise_x = np.random.normal(0, x_sd_noise, len(x_train))
        x_noise = x_train + noise_x

        # Building model
        model = Sequential()
        model.add(Dense(int(input1), activation=dropdown, input_shape=(1,)))
        model.add(Dense(int(input2), activation=dropdown))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mse', 'mape'])
        history = model.fit(x_noise, tg_train,
                            batch_size=100,
                            epochs=int(input3),
                            verbose=1)
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

        traces2.extend([trace_train, trace_test])


        # Metrics
        score_test = model.evaluate(x_test, tg_test, verbose=0)
        score_train = model.evaluate(x_train, tg_train, verbose=0)
        train_mape='Train MAPE: '+str(int(round(score_train[2],0)))+'%'
        test_mape=' Test MAPE: '+str(int(round(score_test[2],0)))+'%'
        train_mse=' Train MSE: '+str(round(score_train[1],4)*100)+'%'
        test_mse=' Test MSE: '+str(round(score_test[1],4)*100)+'%'
        noise_all=' Noise(x,y):('+str(tg_sd_noise)+','+str(x_sd_noise)+')'

    return {'data': traces2, 'layout':  {'title':train_mape+test_mape+train_mse+test_mse+noise_all}}




