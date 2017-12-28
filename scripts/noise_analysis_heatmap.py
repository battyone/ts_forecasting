import plotly as py
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout

# Creating Input & Output
x = np.linspace(0, 3 * np.pi, num=1000)
target = np.sin(x)

np.random.seed(42)

# Ranges of heatmap
x_noise_std_range = np.linspace(0, 0.5, 200)
y_noise_std_range = [0.5]
# y_noise_std_range = np.linspace(0, 0.2, 10)

# score_mse = [[0 for x in range(len(x_noise_std_range))] for y in range(len(y_noise_std_range))]
score_mse = pd.DataFrame(
    data=np.zeros((len(x_noise_std_range), len(y_noise_std_range))),
    index=x_noise_std_range, columns=y_noise_std_range)

for y_i, y_noise_std in enumerate(y_noise_std_range):
    for x_i, x_noise_std in enumerate(x_noise_std_range):
        print('{}/{}: y_noise_std: {}, x_noise_std: {}'.format(x_i+y_i*len(y_noise_std_range) + 1, len(x_noise_std_range) * len(y_noise_std_range),
                                                               y_noise_std, x_noise_std))
        # Introducing noise in output
        noise_tg = np.random.normal(0, y_noise_std, len(x))
        tg_noise = target + noise_tg

        # Splitting data
        x_train, x_test, tg_train, tg_test = train_test_split(x, tg_noise, test_size=0.3, random_state=42)

        # Putting noise into training input
        noise_x = np.random.normal(0, x_noise_std, len(x_train))
        x_noise = x_train + noise_x

        # Neural Network
        model = Sequential()
        model.add(Dense(100, activation='tanh', input_shape=(1,)))
        model.add(Dense(100, activation='tanh'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mse', 'mape'])

        history = model.fit(x_noise, tg_train,
                            batch_size=100,
                            epochs=300,
                            verbose=0)

        # Calculating scores
        # test_mse = str(round(model.evaluate(x_test, tg_test, verbose=0)[1], 4))
        test_mse = model.evaluate(x_test, tg_test, verbose=0)[1]
        train_mse = model.evaluate(x_train, tg_train, verbose=0)[1]

        del model

        # Matrix
        score_mse[y_noise_std][x_noise_std] = test_mse

# Plotting heat map
trace = go.Heatmap(z=score_mse.values, x=score_mse.index, y=score_mse.columns, colorscale='Viridis', reversescale=True)
traces = [trace]
layout = go.Layout(
    title='<b>{}</b>'.format('Heat Map of MSE depending on noise'), titlefont=dict(family='Open Sans', size=20),
    font=dict(family='Open Sans'),
    xaxis=dict(title='<i>{}</i>'.format('x noise std'), titlefont=dict(size=16), tickfont=dict(size=12)),
    yaxis=dict(title='<i>{}</i>'.format('y noise std'), titlefont=dict(size=16), tickfont=dict(size=12)),
    legend=dict(font=dict(size=22), orientation='v')
)
fig = go.Figure(data=traces, layout=layout)
py.offline.plot(fig, filename='../data/output/mse_heatmap.html', auto_open=False)


# Plotting loss chart
traces = []
for y_noise_std in score_mse.columns:
    print(score_mse[y_noise_std].values)
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
py.offline.plot(fig, filename='../data/output/mse__y_noise_std_05.html', auto_open=False)


