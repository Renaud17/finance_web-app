import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA', FutureWarning)
warnings.warn('ARIMA_DEPRECATION_WARN', FutureWarning)
import numpy as np 
import pandas as pd 
from datetime import date
import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('ggplot')
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import yfinance as yf
import requests
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)



class Arima2(object):
    def __init__(self, ticker):
        self.stock = ticker
        self.df = yf.download(self.stock, period='10y', parse_dates=True)
        def get_companyLongName(symbol):
            url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
            result = requests.get(url).json()
            for x in result['ResultSet']['Result']:
                if x['symbol'] == symbol:
                    return x['name']
        self.companyLongName = get_companyLongName(self.stock)


        plt.rcParams["figure.figsize"] = (15,5)
        fig, ax = plt.subplots()
        lag_plot(self.df['Open'], lag=3)
        plt.title(f'{self.companyLongName} ({self.stock}) - Autocorrelation plot with lag = 3', fontsize=30, fontweight='bold')
        plt.xlabel("Date", fontsize=20, fontweight='bold')
        plt.ylabel("price", fontsize=20, fontweight='bold')
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k-', alpha=.75, zorder=0)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	        label.set_fontsize(15)
        ax.grid(True, color='k', linestyle='-', linewidth=1, alpha=.3)
        ax.legend(loc='best',prop={"size":16})        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        
        plt.rcParams["figure.figsize"] = (15,5)
        fig, ax = plt.subplots()

        plt.plot(self.df["Close"])
        plt.title(f'{self.companyLongName} ({self.stock}) - price over time', fontsize=30, fontweight='bold')
        plt.xlabel("time", fontsize=20, fontweight='bold')
        plt.ylabel("price", fontsize=20, fontweight='bold')
        plt.xlim(date(2020,1,1), date(2021,6,1))
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	        label.set_fontsize(15)
        ax.grid(True, color='k', linestyle='-', linewidth=1, alpha=.3)
        ax.legend(loc='best',prop={"size":16})        
        plt.tight_layout()        
        st.pyplot(fig)
        plt.close(fig)


    def runArima(self):
        train_data, test_data = self.df[0:int(len(self.df)*0.7)], self.df[int(len(self.df)*0.7):]
        training_data = train_data['Close'].values
        test_data = test_data['Close'].values
        history = [x for x in training_data]
        model_predictions = []
        N_test_observations = len(test_data)
        for time_point in range(N_test_observations):
            model = ARIMA(history, order=(4,1,0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            model_predictions.append(yhat)
            true_test_value = test_data[time_point]
            history.append(true_test_value)
        MSE_error = mean_squared_error(test_data, model_predictions)
        st.write('Testing Mean Squared Error is {}'.format(MSE_error))

        test_set_range = self.df[int(len(self.df)*0.7):].index

        plt.rcParams["figure.figsize"] = (15,5)
        fig, ax = plt.subplots()
        plt.plot(test_set_range, model_predictions, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
        plt.plot(test_set_range, test_data, color='red', label='Actual Price')
        plt.title(f'{self.companyLongName} ({self.stock}) - Prices Prediction', fontsize=30, fontweight='bold')
        plt.xlabel('Date', fontsize=20, fontweight='bold')
        plt.ylabel('Prices', fontsize=20, fontweight='bold')
        plt.xlim(date(2020,9,1), date(2021,6,1))
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(15)
        ax.grid(True, color='k', linestyle='-', linewidth=1, alpha=.3)
        ax.legend(loc='best',prop={"size":16})        
        plt.tight_layout()
        st.pyplot(fig)



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
        plt.title(f'{self.company} ({self.ticker}) - Prices Prediction', fontsize=30, fontweight='bold')
        plt.xlabel('Date', fontsize=20, fontweight='bold')
        plt.ylabel('Prices', fontsize=20, fontweight='bold')
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	        label.set_fontsize(15)
        ax.grid(True, color='k', linestyle='-', linewidth=1, alpha=.3)
        ax.legend(loc='best',prop={"size":16})        
        plt.tight_layout()  
        plt.grid(True)
        st.pyplot(fig)


#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if __name__ == '__main__':
    Arima2('BA').runArima()



if __name__ == '__main__':
    The_Arima_Model('BA').arima_model()


#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *    