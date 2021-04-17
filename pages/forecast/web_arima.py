import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

import matplotlib
import matplotlib as mpl
matplotlib.use('Agg')
from matplotlib import style
from matplotlib import pyplot as plt
plt.style.use('ggplot')
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
plt.rcParams['figure.dpi'] = 250

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
from datetime import datetime
import yfinance as yf
import streamlit as st


#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


class Web_Arima(object):
    def __init__(self, ticker):
        self.ticker = ticker

        dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
        stock = 'AAPL'
        period = '1y'
        self.start_date = datetime(2019, 1, 1)
        self.end_date = datetime(2020, 12, 30)
        self.data = yf.download(stock, period=period).fillna(0)
        # data = yf.download(stock, start=start_date, end=end_date).fillna(0)

        import requests
        def get_symbol(symbol):
            url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
            result = requests.get(url).json()
            for x in result['ResultSet']['Result']:
                if x['symbol'] == symbol:
                    return x['name']
        self.company = get_symbol(self.ticker)


      #plot close price
        fig, ax = plt.subplots()
        plt.xlabel('Dates')
        plt.ylabel('Close Prices')
        plt.plot(self.data['Close'])
        plt.title(f'{self.company} ({self.ticker}) - closing price')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(True)
        st.pyplot(fig)

        fig, ax = plt.subplots()
        plt.xlabel('Dates')
        plt.ylabel('Close Prices')        
        self.df_close = self.data['Close']
        self.df_close.plot(style='k.')
        plt.title(f'Scatter plot of {self.company} ({self.ticker}) - closing price')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(True)
        st.pyplot(fig)

    #Test for staionarity
    def test_stationarity(self, timeseries):
      #Determing rolling statistics
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).std()

      #Plot rolling statistics:
        fig, ax = plt.subplots()
        plt.xlabel('Dates')
        plt.ylabel('Close Prices')        
        plt.plot(timeseries, color='blue',label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        # plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.title(f'Rolling Mean and Standard Deviation Of {self.company} ({self.ticker})')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(True)
        plt.show(block=False)
        st.pyplot(fig)
        
        st.text("Results of dickey fuller test")
        adft = adfuller(timeseries,autolag='AIC')
      # output for dft will give us without defining what the values are.
      #hence we manually write what values does it explains using a for loop
        output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
        for key,values in adft[4].items():
            output['critical value (%s)'%key] =  values
        st.text(output)


    def full_build(self):
        self.test_stationarity(self.df_close)
        result = seasonal_decompose(self.df_close, model='multiplicative', period = 30)
        fig, ax = plt.subplots()
        fig = result.plot()
        fig.set_size_inches(16, 9)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(True)
        st.pyplot(fig)

        self.df_log = np.log(self.df_close)
        self.moving_avg = self.df_log.rolling(12).mean()
        self.std_dev = self.df_log.rolling(12).std()

        fig, ax = plt.subplots()
        plt.xlabel('Dates')
        plt.ylabel('Close Prices')        
        plt.title(f'Moving Average For {self.company} ({self.ticker})')
        plt.plot(self.df_log, color='green',label = 'Log-Price')
        # plt.plot(self.std_dev, color ="black", label = "Standard Deviation")
        plt.plot(self.moving_avg, color="red", label = "Mean")
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(True)
        st.pyplot(fig)

        #split data into train and training set
        train_data, test_data = self.df_log[3:int(len(self.df_log)*0.8)], self.df_log[int(len(self.df_log)*0.8):]
        fig, ax = plt.subplots()
        plt.xlabel('Dates')
        plt.ylabel(f'Closing Prices For {self.company}')
        plt.plot(self.df_log, 'green', label='Train data')
        plt.plot(test_data, 'blue', label='Test data')
        plt.title(f"{self.company} ({self.ticker}) - Train & Testing Model")
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(True)
        st.pyplot(fig)

        model_autoARIMA = auto_arima(
            train_data, 
            start_p=0, 
            start_q=0,
            test='adf',       # use adftest to find             optimal 'd'
            max_p=3, max_q=3, # maximum p and q
            m=1,              # frequency of series
            d=None,           # let model determine 'd'
            seasonal=False,   # No Seasonality
            start_P=0, 
            D=0, 
            trace=True,
            error_action='ignore',  
            suppress_warnings=True, 
            stepwise=True
            )
        st.text(model_autoARIMA.summary())

        st.pyplot(model_autoARIMA.plot_diagnostics())

        model = ARIMA(train_data, order=(1, 1, 0))  
        fitted = model.fit(disp=-1)  
        st.text(fitted.summary())


        # Forecast
        fc, se, conf = fitted.forecast(51, alpha=0.05)  # 95% confidence
        fc_series = pd.Series(fc, index=test_data.index)
        lower_series = pd.Series(conf[:, 0], index=test_data.index)
        upper_series = pd.Series(conf[:, 1], index=test_data.index)

        fig, ax = plt.subplots()
        plt.plot(train_data, label='training')
        plt.plot(test_data, color = 'blue', label='Actual Stock Price')
        plt.plot(fc_series, color = 'orange',label='Predicted Stock Price')
        plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.10)
        plt.title(f'{self.company} ({self.ticker}) - Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Actual Stock Price')
        plt.legend(loc='best', fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(fig)


#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from datetime import datetime
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import yfinance as yf
from yfinance import ticker


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


class The_Arima_Model(object):
    def __init__(self, ticker, period='1y', interval='1d'):
        self.ticker = ticker
        self.period = period
        self.interval = interval

        import requests
        def get_symbol(symbol):
            url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
            result = requests.get(url).json()
            for x in result['ResultSet']['Result']:
                if x['symbol'] == symbol:
                    return x['name']
        self.company = get_symbol(self.ticker)


    def arima_model(self):
        data = yf.download(self.ticker, period=self.period, interval=self.interval)
        df = pd.DataFrame(data['Close'])
        df.reset_index(inplace=True)
        train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]
        training_data = train_data['Close'].values
        test_data = test_data['Close'].values

        history = [x for x in training_data]
        model_predictions = []
        N_test_observations = len(test_data)

        for time_point in range(N_test_observations):
            model = ARIMA(history, order=(4,1,0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()[0]
            yhat = output[0]
            model_predictions.append(yhat)
            true_test_value = test_data[time_point]
            history.append(true_test_value)

        MSE_error = mean_squared_error(test_data, model_predictions)
        RMSE_error = MSE_error**0.5

        st.write(' * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ')
        st.text(f'\n > {self.ticker}: \n')
        st.text('\nTesting Mean Squared Error is {}'.format(MSE_error))
        st.write('Testing Root Mean Squared Error is {}'.format(RMSE_error),'\n')
        st.text(' * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ')


        df.set_index('Date', inplace=True)
        test_set_range = df[int(len(df)*0.7):].index

        fig, ax = plt.subplots()
        plt.plot(
            test_set_range, 
            model_predictions, 
            color='blue', 
            marker='X', 
            linestyle='--',
            label='Predicted Price'
            )
        plt.plot(test_set_range, test_data, color='red', label='Actual Price')
        plt.title(f'{self.company} ({self.ticker}) - Prices Prediction')
        plt.xlabel('Date')
        plt.ylabel('Prices')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.grid(True)
        st.pyplot(fig)


#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if __name__ == '__main__':
    The_Arima_Model('BA').arima_model()



if __name__ == '__main__':
    Web_Arima('BA').full_build()


#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *