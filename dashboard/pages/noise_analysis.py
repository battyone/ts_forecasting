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
x = np.linspace(-1, 1, num=100)
target = np.sin(3*np.square(x+0.8))
# Splitting data
x_train, x_test, tg_train, tg_test = train_test_split(
    x, target, test_size=0.5, random_state=42)
#
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
                    max=2,
                    value=0,
                    step=0.01,
                    marks={
                        0: {'label': '0'},
                        0.5: {'label': '0.5'},
                        1: {'label': '1'},
                        1.5: {'label': '1.5'},
                        2: {'label': '2'},
                    },
                ),
            ], className='four columns'),
            html.Div([
                dcc.Slider(
                    id='x-noise-slider',
                    min=0,
                    max=0.5,
                    value=0,
                    step=0.01,
                    marks={
                        0: {'label': '0'},
                        0.15: {'label': '0.15'},
                        0.3: {'label': '0.3'},
                        0.5: {'label': '0.5'}
                    },

                ),
            ], className='four columns'),
        ], className='row', style={'margin-bottom': '50px', 'margin-left': '2px'}),
        html.Div([
            html.Div(id='output-container-button_1',
                     children='Neurons in first layer', className='four columns'),
            html.Div(id='output-container-button_2',
                     children='Neurons in second layer', className='four columns'),
            html.Div(id='output-container-button_3',
                     children='Number of epochs', className='four columns')
        ], className='row', style={'margin-bottom': '2px','margin-left': '4px'}),
        html.Div([
            dcc.Input(id='input-1', type='text', value=25, className='four columns'),
            dcc.Input(id='input-2', type='text', value=25, className='four columns'),
            dcc.Input(id='input-3', type='text', value=200, className='four columns')
        ], className='row', style={'margin-bottom': '50px', 'margin-left': '4px'}),
        html.Div([
            html.Div(id='output-container-button_4',
                     children='Generating number', className='four columns')
            ], className='row', style={'margin-bottom': '2px', 'margin-left': '15px'}),
        html.Div([
            dcc.Input(id='n_noise_points', type='text', value=10, className='four columns'),
            dcc.Dropdown(id='dropdown-activation',
                         options=[
                             {'label': 'relu', 'value': 'relu'},
                             {'label': 'tanh', 'value': 'tanh'}
                         ],
                         value='tanh', className='four columns'
                         ),
            html.Button(id='submit-button', n_clicks=0, children='Start NN', className='four columns')
        ], className='row', style={'margin-bottom': '30px', 'margin-left': '10px'}),
        html.Div([
            dcc.Graph(id='loss-function')], className='row')
    ])

    return layout

def gen_xy_noise(x,y, x_sd=0.1,tg_sd=0, n=10):
    x_noise = np.repeat(x, n) + np.random.normal(0, x_sd, n * len(x))
    y_noise = np.repeat(y, n) + np.random.normal(0, tg_sd, n * len(y))
    return x_noise, y_noise

@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    [dash.dependencies.Input('x-noise-slider', 'value'),
    dash.dependencies.Input('target-noise-slider', 'value'),
    dash.dependencies.Input('n_noise_points', 'value')]
)

def update_figure(x_sd_noise,tg_sd_noise, n_noise_points):
    # Introducing noise in output
    np.random.seed(42)

    generated_data = gen_xy_noise(x_train,tg_train, x_sd_noise, tg_sd_noise, int(n_noise_points))

    traces = []

    trace_target = go.Scatter(
        x=x,
        y=target,
        mode='markers',
        name='target')

    trace_noise = go.Scatter(
        x=generated_data[0],
        y=generated_data[1],
        mode='markers',
        name='generated training noise')

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
    dash.dependencies.State('dropdown-activation', 'value'),
    dash.dependencies.State('n_noise_points', 'value')
     ]
)

def update_NN(n_clicks, graph, tg_sd_noise,x_sd_noise, input1, input2, input3, dropdown,n_noise_points):
    traces2 = []
    test_mse=''
    train_mse=''
    train_mape=''
    test_mape=''
    if graph is not None:
        traces2.extend(graph['data'])
    if n_clicks:
        # Introducing noise in output
        np.random.seed(42)
        # noise_tg = np.random.normal(0, tg_sd_noise, len(x))
        # tg_noise = target + noise_tg

        generated_data = gen_xy_noise(x, target, x_sd_noise, tg_sd_noise, int(n_noise_points))

        # Putting noise into training input
        # noise_x = np.random.normal(0, x_sd_noise, len(x_train))
        # x_noise = x_train + noise_x

        # Building model№1 for original data
        model_1 = Sequential()
        model_1.add(Dense(int(input1), activation=dropdown, input_shape=(1,)))
        model_1.add(Dense(int(input2), activation=dropdown))
        model_1.add(Dense(1, activation='linear'))
        model_1.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mse', 'mape'])
        history = model_1.fit(x_train, tg_train,
                            batch_size=32,
                            epochs=int(input3),
                            verbose=1)
        # Comparing results
        NN1_pred_test = model_1.predict(x).reshape(1, -1)
        # NN_pred_train = model.predict(x_train).reshape(1, -1)

        # Building model№2 for generated data
        model_2 = Sequential()
        model_2.add(Dense(int(input1), activation=dropdown, input_shape=(1,)))
        model_2.add(Dense(int(input2), activation=dropdown))
        model_2.add(Dense(1, activation='linear'))
        model_2.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mse', 'mape'])
        history = model_2.fit(generated_data[0], generated_data[1],
                            batch_size=32,
                            epochs=int(input3),
                            verbose=1)
        # Comparing results
        NN2_pred_test = model_2.predict(generated_data[0]).reshape(1, -1)

        trace_test = go.Scatter(
            x=x,
            y=NN1_pred_test[0],
            mode='markers',
            name='NN_1',
            marker=dict(size=10,
                        line=dict(width=3),
                        opacity=0.5
                        ))

        trace_train = go.Scatter(
            x=generated_data[0],
            y=NN2_pred_test[0],
            mode='markers',
            name='NN_2',
            marker=dict(size=5,
                        line=dict(width=1),
                        opacity=0.5
                        ))

        traces2.extend([trace_test,trace_train])


        # Metrics
        score_test_1 = model_1.evaluate(x, target, verbose=0)
        score_train_1 = model_1.evaluate(x_train, tg_train, verbose=0)

        score_test_2 = model_2.evaluate(x, target, verbose=0)
        score_train_2 = model_2.evaluate(x_train, tg_train, verbose=0)

        # train_mse_1 = ' Train MSE_1: ' + str(round(score_train_1[1], 3))
        test_mse_1 = ' Test MSE_1: ' + str(round(score_test_1[1], 3))

        # train_mse_2 = ' Train MSE_2: ' + str(round(score_train_2[1], 3))
        test_mse_2 = ' Test MSE_2: ' + str(round(score_test_2[1], 3))

        noise_all=' Noise(x,y):('+str(tg_sd_noise)+','+str(x_sd_noise)+')'

    return {'data': traces2, 'layout':  {'title':test_mse_1+','+test_mse_2+','+noise_all}}




