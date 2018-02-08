import plotly as py
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

random_number = 11
np.random.seed(random_number)

# Function to approximate
def f(x):
    # y = 0.45*np.sin(x)+0.5
    y = np.sin(3 * np.square(x + 0.8))
    # y = 3*np.sin(x)+1.5*np.sin(4*x)-np.sin(5*x)+2*np.cos(7*x)-1.5*np.sin(10*x)+0.2*np.cos(10*x)-\
    #     0.1*np.cos(15*x)+np.sin(23*x)
    return y


# Parameters
epoch_number = 2
epochs_range = [10, 100]
y_noise_std = 0
x_noise_number = 2
x_noise_range = [0, 0.6]

# Test Data
x = np.linspace(x_range[0], x_range[1], num=1000)

# Ideal target
target = f(x)

# Ranges
x_noise_std_range = np.linspace(x_noise_range[0], x_noise_range[1], x_noise_number)
epoch_range = np.linspace(epochs_range[0], epochs_range[1], epoch_number)

# Score
score_mse = pd.DataFrame(
    data=np.zeros((len(x_noise_std_range), len(epoch_range))),
    index=x_noise_std_range, columns=epoch_range)

def gen_xy_noise(x,y, x_sd=0,tg_sd=0, n=5):
    np.random.seed(random_number)
    x_noise = np.repeat(x, n) + np.random.normal(0, x_sd, n * len(x))
    y_noise = np.repeat(y, n) + np.random.normal(0, tg_sd, n * len(y))
    return x_noise, y_noise


for epoch_i, epoch in enumerate(epoch_range):
    for x_i, x_noise_std in enumerate(x_noise_std_range):
        print('Epoch number: {}, x_noise_std: {}'.format(epoch, x_noise_std))

        # Original Training Data
        x_org = np.linspace(-1, 1, 15)
        target_org = f(x_org) + np.random.normal(0, 0.4, len(x_org))

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
                              epochs=int(epoch),
                              verbose=1)
        NN2_pred = model_2.predict(x).reshape(1, -1)

        # Calculating scores
        # test_mse = str(round(model.evaluate(x_test, tg_test, verbose=0)[1], 4))
        test_mse = model_2.evaluate(x, target, verbose=0)[1]
        train_mse = model_2.evaluate(x_org, target_org, verbose=0)[1]

        del model_2

        # Matrix
        score_mse[epoch][x_noise_std] = test_mse
        # print('epoch: {}'.format(score_mse.columns))
        # print('x_noise_std: {}'.format(score_mse.index))
        # print('Epoch: {}'.format(epoch))
        # print('MSE value: {}'.format(list(score_mse[epoch].values)))

# Plotting loss chart
traces = []
for epoch in score_mse.columns:
    trace = go.Scatter(x=list(score_mse.index), y=score_mse[epoch].values,
                   name='Number of Epochs: {}'.format(epoch))
    traces.append(trace)

layout = go.Layout(
    title='<b>{}</b>'.format('MSE for y_noise_std: {}'.format(y_noise_std)),
    titlefont=dict(family='Open Sans', size=20),
    font=dict(family='Open Sans'),
    xaxis=dict(title='<i>{}</i>'.format('x noise std'), titlefont=dict(size=16), tickfont=dict(size=12)),
    yaxis=dict(title='<i>{}</i>'.format('MSE'), titlefont=dict(size=16), tickfont=dict(size=12)),
    legend=dict(font=dict(size=16), orientation='v'),
)

fig = go.Figure(data=traces, layout=layout)
py.offline.plot(fig,
                filename='/Users/stepazar/Library/Mobile Documents/com~apple~CloudDocs/Presentation arhive/mse__y_noise_std_05.html',
                auto_open=False)


