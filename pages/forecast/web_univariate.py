#   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, date, timedelta
from pathlib import Path
today = str(datetime.now())[:10]

import matplotlib
import matplotlib as mpl
matplotlib.use('Agg')
from matplotlib import style
from matplotlib import pyplot as plt
plt.style.use('seaborn-talk')
# plt.style.use('seaborn-poster')
# plt.style.use('_classic_test_patch')
# plt.style.use('fast')
# plt.style.use('fivethirtyeight')
# plt.style.use('seaborn-dark-palette')
# plt.style.use('seaborn-colorblind')
# plt.style.use('seaborn-deep')
# plt.style.use('seaborn-muted')
# plt.style.use('seaborn-notebook')
# plt.style.use('ggplot')
sm, med, lg = 10, 15, 20
plt.rc('font', size = sm)         # controls default text sizes
plt.rc('axes', titlesize = med)   # fontsize of the axes title
plt.rc('axes', labelsize = med)   # fontsize of the x & y labels
plt.rc('xtick', labelsize = sm)   # fontsize of the tick labels
plt.rc('ytick', labelsize = sm)   # fontsize of the tick labels
plt.rc('legend', fontsize = sm)   # legend fontsize
plt.rc('figure', titlesize = lg)  # fontsize of the figure title
plt.rc('axes', linewidth=2)       # linewidth of plot lines
plt.rcParams['figure.figsize'] = [18, 10]
plt.rcParams['figure.dpi'] = 150

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import itertools
import streamlit as st
from prophet.plot import plot_cross_validation_metric
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.plot import add_changepoints_to_plot
from prophet import Prophet
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow import keras

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    tf.config.experimental.set_synchronous_execution(enable=True)
    tf.config.experimental.enable_mlir_bridge
    tf.config.experimental.enable_tensor_float_32_execution(enabled=True)
    tf.config.threading.get_inter_op_parallelism_threads()
    tf.config.threading.set_inter_op_parallelism_threads(0)
else:
    print('Using CPU')


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


class The_Univariate_TS_Reg(object):
    def __init__(self, stock_symbol):
        self.ticker = stock_symbol
        self.saver = stock_symbol

        import requests
        def get_symbol(symbol):
            url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
            result = requests.get(url).json()
            for x in result['ResultSet']['Result']:
                if x['symbol'] == symbol:
                    return x['name']
        self.company = get_symbol(self.ticker)        

    def runs(self):
        if self.ticker:
            sp500 = yf.download(self.ticker, period='5y', interval='1d')
            sp500 = pd.DataFrame(sp500['Adj Close'])
            sp500.columns = [self.saver]
            sp500.fillna(0.0, inplace=True)
            scaler = MinMaxScaler()
            sp500_scaled = pd.Series(scaler.fit_transform(sp500).squeeze(), index=sp500.index)
            st.write(sp500_scaled.describe())

            def create_univariate_rnn_data(data, window_size):
                n = len(data)
                y = data[window_size:]
                data = data.values.reshape(-1, 1) # make 2D
                X = np.hstack(tuple([data[i: n-j, :] for i, j in enumerate(range(window_size, 0, -1))]))
                return pd.DataFrame(X, index=y.index), y

            window_size = 63
            X, y = create_univariate_rnn_data(sp500_scaled, window_size)
            X_train = X[:'2020'].values.reshape(-1, window_size, 1)
            y_train = y[:'2020']
        # keep the last year for testing
            X_test = X['2020':].values.reshape(-1, window_size, 1)
            y_test = y['2020':]
            n_obs, window_size, n_features = X_train.shape

            rnn = Sequential([LSTM(
                units=10, 
                input_shape=(window_size, n_features),
                name='LSTM'),
                Dense(n_features, name='Output'),
                ])
            st.text(rnn.summary())

            rnn.compile(loss='mae', optimizer='RMSProp')
            
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=10,
                restore_best_weights=True
                )
            lstm_training = rnn.fit(
                X_train,
                y_train,
                epochs=100,
                batch_size=40,
                shuffle=True,
                validation_data=(X_test, y_test),
                verbose=1
                )

            fig, ax = plt.subplots()
            loss_history = pd.DataFrame(lstm_training.history).pow(.5)
            loss_history.index += 1
            best_rmse = loss_history.val_loss.min()
            best_epoch = loss_history.val_loss.idxmin()
            title = f'5-Epoch Rolling RMSE (Best Validation RMSE: {best_rmse:.4%})'
            loss_history.columns=['Training RMSE', 'Validation RMSE']
            loss_history.rolling(5).mean().plot(logy=True, lw=2, title=title, ax=ax)
            ax.axvline(best_epoch, ls='--', lw=1, c='k')
            sns.despine()
            plt.grid(True)
            fig.tight_layout()
            plt.legend(loc='best')
            st.pyplot(fig)
            plt.close(fig)
            
            train_rmse_scaled = np.sqrt(rnn.evaluate(X_train, y_train, verbose=1))
            test_rmse_scaled = np.sqrt(rnn.evaluate(X_test, y_test, verbose=1))
            st.write(f'Train RMSE: {train_rmse_scaled:.4} | Test RMSE: {test_rmse_scaled:.4}')

            train_predict_scaled = rnn.predict(X_train)
            test_predict_scaled = rnn.predict(X_test)

            train_ic = spearmanr(y_train, train_predict_scaled)[0]
            test_ic = spearmanr(y_test, test_predict_scaled)[0]
            st.write(f'Train IC: {train_ic} | Test IC: {test_ic}')

            train_predict = (pd.Series(scaler.inverse_transform(train_predict_scaled).squeeze(), index=y_train.index))
            test_predict = (pd.Series(scaler.inverse_transform(test_predict_scaled).squeeze(), index=y_test.index))

            y_train_rescaled = scaler.inverse_transform(y_train.to_frame()).squeeze()
            y_test_rescaled = scaler.inverse_transform(y_test.to_frame()).squeeze()

            train_rmse = np.sqrt(mean_squared_error(train_predict, y_train_rescaled))
            test_rmse = np.sqrt(mean_squared_error(test_predict, y_test_rescaled))
            st.write(f'Train RMSE: {train_rmse:.2} | Test RMSE: {test_rmse:.2}')

            sp500['Train Predictions'] = train_predict
            sp500['Test Predictions'] = test_predict
            sp500 = sp500.join(
                train_predict.to_frame('predictions').assign(data='Train').append(
                    test_predict.to_frame('predictions').assign(data='Test')))

            fig=plt.figure()
            ax1 = plt.subplot(221)
            sp500.loc['2018':, self.saver].plot(lw=3, ax=ax1, c='k', alpha=.6)
            sp500.loc['2018':, ['Test Predictions', 'Train Predictions']].plot(lw=2, ax=ax1, ls='--')
            ax1.set_title(f'In- and Out-of-sample Predictions For ~ {self.saver}')
            ax1.set_ylabel('Stock Price')
            with sns.axes_style("white"):
                ax3 = plt.subplot(223)
                sns.scatterplot(x=self.saver, y='predictions', data=sp500, hue='data', ax=ax3)
                ax3.text(x=.02, y=.95, s=f'Test IC ={test_ic:.2%}', transform=ax3.transAxes)
                ax3.text(x=.02, y=.90, s=f'Train IC={train_ic:.2%}', transform=ax3.transAxes)
                ax3.set_title('Correlation Plot ~ ')
                ax3.legend(loc='lower right')
                ax2 = plt.subplot(222)
                ax4 = plt.subplot(224, sharex = ax2, sharey=ax2)
                sns.distplot(train_predict.squeeze()- y_train_rescaled, ax=ax2)
                ax2.set_title('Train Error')
                ax2.text(x=.03, y=.92, s=f'Train RMSE ={train_rmse:.4f}', transform=ax2.transAxes)
                ax2.set_ylabel('val_loss - Train_Tally')
                ax2.set_xlabel('Root_Mean_Squared_Error - Train_Tally')
                sns.distplot(test_predict.squeeze()-y_test_rescaled, ax=ax4)
                ax4.set_title('Test Error')
                ax4.text(x=.03, y=.92, s=f'Test RMSE ={test_rmse:.4f}', transform=ax4.transAxes)
                ax4.set_ylabel('val_loss - Test_Tally (less train) = Final')
                ax4.set_xlabel('Root_Mean_Squared_Error - Test_Tally >>> Final')            
            sns.despine()
            plt.title(f'Univariate Model of {self.company} ({self.saver})')
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(fig)

            fig, ax = plt.subplots()    
            ax = sp500.loc['2018':, self.saver].plot(lw=2, c='k', alpha=.6, label=f'{self.saver} Observed_Stock_Price')
            sp500.loc['2019':, ['Test Predictions', 'Train Predictions']].plot(ax=ax, lw=1, style=[':','--'])
            ax.vlines(['2017-07-01', '2018-12-25'] ,0, 1, transform=ax.get_xaxis_transform(), colors='r', lw=2, ls='--', label='Train_Range')
            ax.vlines(['2019-01-01', '2019-12-25'] ,0, 1, transform=ax.get_xaxis_transform(), colors='y', lw=2, ls='--', label='Validation_Range')
            ax.vlines(['2020-01-01', '2020-12-25'] ,0, 1, transform=ax.get_xaxis_transform(), colors='g', lw=2, ls='--', label='Test_Prediction_Range')
            ax.vlines(['2021-01-01', '2021-06-30'] ,0, 1, transform=ax.get_xaxis_transform(), colors='b', lw=2, ls='--', label='Out_Of_Sample-Prediction_Range')
            ax.set_title(f'In- and Out-of-sample Predictions For ~ {self.company} ({self.saver})')
            ax.set_ylabel('Stock Price ($)')
            ax.set_xlabel('Date')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(fig)
            return


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *        


if __name__ == '__main__':
    The_Univariate_TS_Reg('TSLA').runs()


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *    