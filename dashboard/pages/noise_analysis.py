import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input
from keras.layers import Dense
from keras.models import Sequential
from dashboard.app import app
# from keras import regularizers
# from sklearn.model_selection import train_test_split
# from keras import optimizers


random_number=11
# np.random.randint(1,100)
# print(random_number)


# Function to approximate
def f(x):
    y = 0.4*np.sin(x)+0.5
    # y = np.sin(3 * np.square(x + 0.8))
    # y = 3*np.sin(x)+1.5*np.sin(4*x)-np.sin(5*x)+2*np.cos(7*x)-1.5*np.sin(10*x)+0.2*np.cos(10*x)-\
    #     0.1*np.cos(15*x)+np.sin(23*x)
    return y

# Parameters
# x_range=[-1,1]
x_range=[-np.pi,np.pi]
y_measurement_error=0.1
# y_measurement_error=0.4

# Test Data
x = np.linspace(x_range[0], x_range[1], num=1000)

# Ideal target
target = f(x)

# Design
def render_noise_analysis_layout():
    header = html.Div(
        className='header',
        children=[
            html.A([html.Img(src='/static/back_arrow.png', style={'height': '40px', 'margin-right': '470px'}),
                    html.Div('',
                             style={'font-size': '16px', 'font-weight': '500',
                                    'font-family': 'Open Sans', 'color': "#fff", 'letter-spacing': '0.470px',
                                    'text-transform': 'uppercase'})
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
                               'margin-bottom': '470px'
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
                dcc.Graph(id='graph1')], className='six columns'),
            html.Div([
                dcc.Graph(id='graph2')], className='six columns')
            ], className='row'),
        html.Div([
                html.Div(id='output-a', className='four columns'),
                html.Div(id='output-b', className='four columns')
        ], className='row', style={'margin-bottom': '4px', 'margin-left': '470px'}),
        html.Div([
            html.Div([
                dcc.Slider(
                    id='target-noise-slider',
                    min=0,
                    max=0.5,
                    value=0,
                    step=0.01,
                    marks={
                        0: {'label': '0'},
                        0.15: {'label': '0.15'},
                        0.3: {'label': '0.3'},
                        0.5: {'label': '0.5'}
                    }
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
            ], className='four columns')
        ], className='row', style={'margin-bottom': '20px', 'margin-left': '470px'}),
        html.Div([
                html.Div(id='training_split', className='four columns'),
                html.Div(id='output-container-button_4', children='Generating number', className='four columns')
        ], className='row', style={'margin-bottom': '0px', 'margin-left': '470px'}),
        html.Div([
                dcc.Input(id='data', type='text', value=15, className='four columns'),
                dcc.Input(id='n_noise_points', type='text', value=5, className='four columns')
        ], className='row', style={'margin-bottom': '20px', 'margin-left': '470px'}),
        html.Div([
            html.Div(id='output-container-button_1',
                     children='Neurons in the first layer', className='four columns'),
            html.Div(id='output-container-button_3',
                     children='Number of epochs', className='four columns')
        ], className='row', style={'margin-bottom': '0px','margin-left': '470px'}),
        html.Div([
            dcc.Input(id='input-1', type='text', value=15, className='four columns'),
            dcc.Input(id='input-3', type='text', value=500, className='four columns')
        ], className='row', style={'margin-bottom': '20px', 'margin-left': '470px'}),
        html.Div([
            html.Div(id='output-container-button_2',
                     children='Activation function (first layer)', className='four columns'),
            html.Div(id='output-container-button_4',
                     children='Activation function (output layer)', className='four columns')
        ], className='row', style={'margin-bottom': '0px', 'margin-left': '470px'}),
        html.Div([
            html.Div([
                dcc.Dropdown(id='dropdown-activation',
                             options=[
                                 {'label': 'relu', 'value': 'relu'},
                                 {'label': 'tanh', 'value': 'tanh'},
                                 {'label': 'sigmoid', 'value': 'sigmoid'}
                             ],
                             value='tanh', className='four columns')], style = {}),
            html.Div([
                dcc.Dropdown(id='dropdown-activation-output',
                             options=[
                                 {'label': 'linear', 'value': 'linear'},
                                 {'label': 'sigmoid', 'value': 'sigmoid'}
                             ],
                             value='linear', className='four columns',
                             )], style = {'margin-left': '500px'})
        ], className='row', style={'margin-bottom': '20px', 'margin-left': '470px'}),
        html.Div([
            html.Button(id='submit-button', n_clicks=0, children='Start NN', className='row')
        ], className='row', style={'margin-bottom': '20px', 'margin-left': '885px'})
    ])
    return layout


def gen_xy_noise(x,y, x_sd=0,tg_sd=0, n=5):
    np.random.seed(random_number)
    x_noise = np.repeat(x, n) + np.random.normal(0, x_sd, n * len(x))
    y_noise = np.repeat(y, n) + np.random.normal(0, tg_sd, n * len(y))
    return x_noise, y_noise


@app.callback(
    dash.dependencies.Output('output-a', 'children'),
    [dash.dependencies.Input('target-noise-slider', 'value')])
def callback_a(value):
    return 'Generated Additive Noise on Y ~ N(0,{})'.format(np.square(value).round(3))

@app.callback(
    dash.dependencies.Output('output-b', 'children'),
    [dash.dependencies.Input('x-noise-slider', 'value')])
def callback_a(value):
    return 'Generated Additive Noise on X ~ N(0,{})'.format(np.square(value).round(3))

@app.callback(
    dash.dependencies.Output('training_split', 'children'),
    [dash.dependencies.Input('data', 'value')])
def callback_a(value):
    return 'Training set proportion: {}%'.format(float(value)/len(x)*100)


@app.callback(
    dash.dependencies.Output('graph1', 'figure'),
    [dash.dependencies.Input('x-noise-slider', 'value'),
    dash.dependencies.Input('target-noise-slider', 'value'),
     dash.dependencies.Input('data', 'value'),
    dash.dependencies.Input('n_noise_points', 'value')
     ]
)

def update_figure(x_sd_noise,tg_sd_noise, data, n_noise_points):
    np.random.seed(random_number)
    # Splitting data
    # x_train, x_test, tg_train, tg_test = train_test_split(
    #     x, target, test_size=(1-float(data)/len(x)), random_state=random_number)

    # Training data
    x_org = np.linspace(x_range[0], x_range[1], num=int(data))
    target_org = f(x_org) + np.random.normal(0, y_measurement_error, len(x_org))

    # Generated data
    generated_data = gen_xy_noise(x_org, target_org, x_sd_noise, tg_sd_noise, int(n_noise_points))

    # print('x_gen_noise: {}, y_gen_noise: {}'.format(len(generated_data[0]), len(generated_data[1])))
    # print('x_train[0]: {}'.format(x_train[0]))
    # print('n_noise_points: {}'.format(n_noise_points))
    # print('generated_data[0]: {}'.format(generated_data[0]))

    traces = []

    trace_target = go.Scatter(
        x=x,
        y=target,
        mode='markers',
        name='Function to approximate')

    trace_noise = go.Scatter(
        x=generated_data[0],
        y=generated_data[1],
        mode='markers',
        name='Generated Data for Training')

    trace_train = go.Scatter(
        x=x_org,
        y=target_org,
        mode='markers',
        name='Original Data for Training')

    traces.extend([trace_target, trace_noise,trace_train])

    return {'data': traces}

@app.callback(
    dash.dependencies.Output('graph2', 'figure'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('graph1', 'figure'),
    dash.dependencies.State('target-noise-slider', 'value'),
    dash.dependencies.State('x-noise-slider', 'value'),
    dash.dependencies.State('data', 'value'),
    dash.dependencies.State('input-1', 'value'),
    dash.dependencies.State('input-3', 'value'),
    dash.dependencies.State('dropdown-activation', 'value'),
    dash.dependencies.State('dropdown-activation-output', 'value'),
    dash.dependencies.State('n_noise_points', 'value')
     ]
)

def update_NN(n_clicks, graph1, tg_sd_noise,x_sd_noise, data, input1, input3, dropdown, dropdown_output, n_noise_points):
    np.random.seed(random_number)
    traces2 = []
    mse_1 = ''
    mse_2 = ''
    noise_all = ''
    if n_clicks:
        # Splitting data
        # x_train, x_test, tg_train, tg_test = train_test_split(
        #     x, target, test_size=(1-float(data)/len(x)), random_state=random_number)

        # Training Data
        x_org = np.linspace(x_range[0], x_range[1], num=int(data))
        target_org = f(x_org) + np.random.normal(0, 0.4, len(x_org))

        # Generated Training Data
        generated_data = gen_xy_noise(x_org, target_org, x_sd_noise, tg_sd_noise, int(n_noise_points))

        # Building model №1 for original data
        model_1 = Sequential()
        model_1.add(Dense(int(input1), activation=dropdown, input_shape=(1,)))
        model_1.add(Dense(1, activation=dropdown_output))
        # sgd = optimizers.SGD(lr=0.01, momentum=0.1, decay=0.0, nesterov=False)
        model_1.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mse'])
        weights=model_1.get_weights()
        # print('Weights: {}'.format(weights))
        history = model_1.fit(x_org, target_org,
                            batch_size=len(x_org),
                            epochs=int(input3),
                            verbose=1)
        NN1_pred = model_1.predict(x).reshape(1, -1)

        # Building model №2 for generated data
        model_2 = Sequential()
        model_2.add(Dense(int(input1), activation=dropdown, input_shape=(1,)))
        model_2.add(Dense(1, activation=dropdown_output))
        # sgd=optimizers.SGD(lr=0.01, momentum=0.1, decay=0.0, nesterov=False)
        model_2.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mse'])
        model_2.set_weights(weights)
        # print(model_2.get_weights())
        history = model_2.fit(generated_data[0], generated_data[1],
                            batch_size=len(generated_data[0]),
                            epochs=int(input3),
                            verbose=1)
        NN2_pred = model_2.predict(x).reshape(1, -1)

        # Comparing results
        trace_orig = go.Scatter(
            x=x,
            y=NN1_pred[0],
            mode='markers',
            name='NN_original')

        trace_gen = go.Scatter(
            x=x,
            y=NN2_pred[0],
            mode='markers',
            name='NN_generated',
            marker=dict(size=5,
                        line=dict(width=1),
                        opacity=0.5
                        ))

        # Metrics
        score_1 = model_1.evaluate(x, target, verbose=0)
        score_2 = model_2.evaluate(x, target, verbose=0)
        mse_1 = 'MSE 1: {}||'.format(str(round(score_1[1], 3)))
        mse_2 = 'MSE 2: {}||'.format(str(round(score_2[1], 3)))
        noise_all='Noise on X & Y ~ N(0,{}) & N(0,{})'.format(np.square(x_sd_noise).round(3),
                                                       np.square(tg_sd_noise).round(3))
        if graph1 is not None:
            traces2.extend(graph1['data'])

        traces2.extend([trace_orig, trace_gen])

    return {'data': traces2, 'layout':  {'title': mse_1+mse_2+noise_all}}