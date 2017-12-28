import plotly as py
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout

def gen_xy_noise(x,y, x_sd=0,tg_sd=0, n=5):
    np.random.seed(random_number)
    x_noise = np.repeat(x, n) + np.random.normal(0, x_sd, n * len(x))
    y_noise = np.repeat(y, n) + np.random.normal(0, tg_sd, n * len(y))
    return x_noise, y_noise

# Create ideal function

x = np.linspace(-1, 1, num=100)
target = np.sin(3 * np.square(x + 0.8))

# Ranges of heatmap
x_noise_std_range = np.linspace(0, 0.2, 5)
random_number_range =np.linspace(1, 3, 3)
# y_noise_std_range = [0.5]
# y_noise_std_range = np.linspace(0, 0.2, 10)

# score_mse = [[0 for x in range(len(x_noise_std_range))] for y in range(len(y_noise_std_range))]
score_mse = pd.DataFrame(
    data=np.zeros((len(x_noise_std_range), len(random_number_range))),
    index=x_noise_std_range, columns=random_number_range)

for  x_i, x_noise_std in enumerate(x_noise_std_range):
    for random_i, random_number in enumerate(random_number_range):
        print('{}/{}: random_number: {}, x_noise_std: {}'.format(x_i+random_i*len(random_number_range) + 1, len(x_noise_std_range) * len(random_number_range),
                                                                 random_number, x_noise_std))
        # Generated train data
        np.random.seed(random_number)
        x_org = np.linspace(-1, 1, num=15)
        target_org = np.sin(3 * np.square(x_org + 0.8)) + np.random.normal(0, 0.4, len(x_org))
        generated_data = gen_xy_noise(x_org, target_org, x_sd_noise, tg_sd_noise, int(5))

        # Building model №1 for original data
        model_1 = Sequential()
        model_1.add(Dense(15, activation=dropdown, input_shape=(1,)))
        # model_1.add(Dense(int(input2), activation=dropdown))
        model_1.add(Dense(1, activation='linear'))
        model_1.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mse', 'mape'])
        weights=model_1.get_weights()
        # print('Weights: {}'.format(weights))
        history = model_1.fit(x_org, target_org,
                            batch_size=len(x_org),
                            epochs=int(input3),
                            verbose=1)
        NN1_pred = model_1.predict(x).reshape(1, -1)

        # Building model №2 for generated data
        model_2 = Sequential()
        model_2.add(Dense(15, activation=dropdown, input_shape=(1,)))
        # model_2.add(Dense(int(input2), activation=dropdown))
        model_2.add(Dense(1, activation='linear'))
        model_2.compile(loss='mse',
                      optimizer='adam',
                      metrics=['mse', 'mape'])
        model_2.set_weights(weights)
        # print(model_2.get_weights())
        history = model_2.fit(generated_data[0], generated_data[1],
                            batch_size=len(x_org),
                            epochs=10,
                            verbose=1)
        NN2_pred = model_2.predict(x).reshape(1, -1)

        # Calculating scores
        mse_1=model_1.evaluate(x, target, verbose=0)[1]
        mse_2=model_2.evaluate(x, target, verbose=0)[1]

        del model

        # Matrix
        score_mse_1[random_number][x_noise_std] = mse_1
        score_mse_2[random_number][x_noise_std] = mse_1

# Plotting heat map
trace = go.Heatmap(z=score_mse_1.values, x=score_mse_1.index, y=score_mse_1.columns, colorscale='Viridis', reversescale=True)
traces = [trace]
layout = go.Layout(
    title='<b>{}</b>'.format('Heat Map of MSE depending on noise'), titlefont=dict(family='Open Sans', size=20),
    font=dict(family='Open Sans'),
    xaxis=dict(title='<i>{}</i>'.format('x noise std'), titlefont=dict(size=16), tickfont=dict(size=12)),
    yaxis=dict(title='<i>{}</i>'.format('y noise std'), titlefont=dict(size=16), tickfont=dict(size=12)),
    legend=dict(font=dict(size=22), orientation='v'),
)
fig = go.Figure(data=traces, layout=layout)
py.offline.plot(fig, filename='../data/output/mse_heatmap.html', auto_open=False)

# Plotting heat map
trace = go.Heatmap(z=score_mse_2.values, x=score_mse_2.index, y=score_mse_2.columns, colorscale='Viridis', reversescale=True)
traces = [trace]
layout = go.Layout(
    title='<b>{}</b>'.format('Heat Map of MSE depending on noise'), titlefont=dict(family='Open Sans', size=20),
    font=dict(family='Open Sans'),
    xaxis=dict(title='<i>{}</i>'.format('x noise std'), titlefont=dict(size=16), tickfont=dict(size=12)),
    yaxis=dict(title='<i>{}</i>'.format('y noise std'), titlefont=dict(size=16), tickfont=dict(size=12)),
    legend=dict(font=dict(size=22), orientation='v'),
)
fig = go.Figure(data=traces, layout=layout)
py.offline.plot(fig, filename='../data/output/mse_heatmap.html', auto_open=False)


# # Plotting loss chart
# traces = []
# for y_noise_std in score_mse.columns:
#     print(score_mse[y_noise_std].values)
#     trace = go.Scatter(x=list(score_mse.index), y=score_mse[y_noise_std].values, name='y_noise_std: {}'.format(y_noise_std))
#     traces.append(trace)
# layout = go.Layout(
#     title='<b>{}</b>'.format('MSE for y_noise_std: {}'.format(y_noise_std_range)), titlefont=dict(family='Open Sans', size=20),
#     font=dict(family='Open Sans'),
#     xaxis=dict(title='<i>{}</i>'.format('x noise std'), titlefont=dict(size=16), tickfont=dict(size=12)),
#     yaxis=dict(title='<i>{}</i>'.format('MSE'), titlefont=dict(size=16), tickfont=dict(size=12)),
#     legend=dict(font=dict(size=16), orientation='v'),
# )
#
# fig = go.Figure(data=traces, layout=layout)
# py.offline.plot(fig, filename='../data/output/mse__y_noise_std_05.html', auto_open=False)


