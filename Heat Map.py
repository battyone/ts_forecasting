import plotly as py
import plotly.graph_objs as go
import numpy as np
from __future__ import print_function
py.offline.init_notebook_mode(connected=True)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

# Creating Input & Output
x = np.linspace(0, 3*np.pi, num=1000)
target = np.sin(x)

np.random.seed(42)

# Ranges of heatmap
y1=0
yn=10
x1=0
xn=20

score_mse = [[0 for x in range(xn)] for y in range(yn)]

for tg_sd_noise in range(y1,yn):
    for x_sd_noise in range(x1,xn):
        # Introducing noise in output
        noise_tg = np.random.normal(0, tg_sd_noise, len(x))
        tg_noise = target + noise_tg

        # Splitting data
        x_train, x_test, tg_train, tg_test = train_test_split(
            x, tg_noise, test_size=0.3, random_state=42)

        # Putting noise into training input
        noise_x = np.random.normal(0, x_sd_noise, len(x_train))
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
                            epochs=1,
                            verbose=0)

        # Calculating scores
        test_mse = str(round(model.evaluate(x_test, tg_test, verbose=0)[1],4)*100)
        train_mse = str(round(model.evaluate(x_train, tg_train, verbose=0)[1],4)*100)

        # Matrix
        score_mse[tg_sd_noise][x_sd_noise]=test_mse

# Plotting heat map
trace = go.Heatmap(z=score_mse)
layout = go.Layout(title = 'Heat Map of MSE depending on noise')
data=[trace]
fig = go.Figure(data=[trace],
                layout=layout)
py.offline.iplot(fig)