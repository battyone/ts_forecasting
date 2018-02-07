import plotly as py
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

random_number=11

# Function to approximate
def f(x):
    y = 0.45*np.sin(x)+0.5
    # y = np.sin(3 * np.square(x + 0.8))
    # y = 3*np.sin(x)+1.5*np.sin(4*x)-np.sin(5*x)+2*np.cos(7*x)-1.5*np.sin(10*x)+0.2*np.cos(10*x)-\
    #     0.1*np.cos(15*x)+np.sin(23*x)
    return y


# Parameters
x_noise_number=2
x_noise_range = [0,0.6]
y_noise_number=2
y_noise_range = [0, 0.6]
epoch_number=100

# Other
y_measurement_error=0.4
x = np.linspace(-1, 1, num=1000)
x_range=[-1,1]
# x_range=[-np.pi,np.pi]
y_measurement_error=0.4

# Test Data
x = np.linspace(x_range[0], x_range[1], num=1000)

# Ideal target
target = f(x)


# Ranges
x_noise_std_range = np.linspace(x_noise_range[0], x_noise_range[1], x_noise_number)
y_noise_std_range = np.linspace(y_noise_range[0], y_noise_range[1], y_noise_number)
# y_noise_std_range = [0.5]

score_mse = pd.DataFrame(
    data=np.zeros((len(x_noise_std_range), len(y_noise_std_range))),
    index=x_noise_std_range, columns=y_noise_std_range)

def gen_xy_noise(x,y, x_sd=0,tg_sd=0, n=5):
    np.random.seed(random_number)
    x_noise = np.repeat(x, n) + np.random.normal(0, x_sd, n * len(x))
    y_noise = np.repeat(y, n) + np.random.normal(0, tg_sd, n * len(y))
    return x_noise, y_noise

for y_i, y_noise_std in enumerate(y_noise_std_range):
    for x_i, x_noise_std in enumerate(x_noise_std_range):
        print('{}/{}: y_noise_std: {}, x_noise_std: {}'.format(x_i+y_i*len(y_noise_std_range) + 1, len(x_noise_std_range) * len(y_noise_std_range),
                                                               y_noise_std, x_noise_std))

        # Original Training Data
        np.random.seed(random_number)
        x_org = np.linspace(-1, 1, 15)
        target_org = f(x_org) + np.random.normal(0, y_measurement_error, len(x_org))

        # Generated Training Data
        generated_data = gen_xy_noise(x_org, target_org, x_noise_std, y_noise_std, 1000)

        # Building model №2 for generated data
        model_2 = Sequential()
        model_2.add(Dense(15, activation='tanh', input_shape=(1,)))
        model_2.add(Dense(1, activation='linear'))
        model_2.compile(loss='mse',
                        optimizer='adam',
                        metrics=['mse'])
        history = model_2.fit(generated_data[0], generated_data[1],
                              batch_size=len(generated_data[0]),
                              epochs=epoch_number,
                              verbose=1)
        NN2_pred = model_2.predict(x).reshape(1, -1)

        # Calculating scores
        # test_mse = str(round(model.evaluate(x_test, tg_test, verbose=0)[1], 4))
        test_mse = model_2.evaluate(x, target, verbose=0)[1]
        train_mse = model_2.evaluate(x_org, target_org, verbose=0)[1]

        del model_2

        # Matrix
        score_mse[y_noise_std][x_noise_std] = test_mse
        print(score_mse.columns)
        print(score_mse.index)
        # print(list(score_mse.columns.values))

# Plotting heat map
trace = go.Heatmap(z=score_mse.values, x=score_mse.index, y=score_mse.columns, colorscale='Viridis', reversescale=True,
                   transpose = True)
traces = [trace]
layout = go.Layout(
    title='<b>{}</b>'.format('Heat Map of MSE depending on noise'), titlefont=dict(family='Open Sans', size=20),
    font=dict(family='Open Sans'),
    xaxis=dict(title='<i>{}</i>'.format('x noise std'), titlefont=dict(size=16), tickfont=dict(size=12)),
    yaxis=dict(title='<i>{}</i>'.format('y noise std'), titlefont=dict(size=16), tickfont=dict(size=12)),
    legend=dict(font=dict(size=22), orientation='v')
)
fig = go.Figure(data=traces, layout=layout)
py.offline.plot(fig, filename='/Users/stepazar/Library/Mobile Documents/com~apple~CloudDocs/Presentation arhive/mse_heatmap.html', auto_open=False)


# Plotting loss chart
traces = []
for y_noise_std in score_mse.columns:
    trace = go.Scatter(x=list(score_mse.index), y=score_mse[y_noise_std].values, name='y_noise_std: {}'.format(y_noise_std))
    traces.append(trace)
layout = go.Layout(
    title='<b>{}</b>'.format('MSE for y_noise_std: {}'.format(y_noise_std_range)), titlefont=dict(family='Open Sans', size=20),
    font=dict(family='Open Sans'),
    xaxis=dict(title='<i>{}</i>'.format('x noise std'), titlefont=dict(size=16), tickfont=dict(size=12)),
    yaxis=dict(title='<i>{}</i>'.format('MSE'), titlefont=dict(size=16), tickfont=dict(size=12)),
    legend=dict(font=dict(size=16), orientation='v'),
)

fig = go.Figure(data=traces, layout=layout)
py.offline.plot(fig, filename='/Users/stepazar/Library/Mobile Documents/com~apple~CloudDocs/Presentation arhive/mse__y_noise_std_05.html', auto_open=False)


