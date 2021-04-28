import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, date, timedelta
from pathlib import Path
today = str(datetime.now())[:10]
import pandas as pd
pd.plotting.register_matplotlib_converters()
import numpy as np
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
# matplotlib.use('Agg')
from matplotlib import style
from matplotlib import pyplot as plt
plt.style.use('ggplot')
sm, med, lg = '20', '25', '30'
plt.rcParams['font.size'] = sm  # controls default text sizes
plt.rc('axes', titlesize = med)   # fontsize of the axes title
plt.rc('axes', labelsize = med)   # fontsize of the x & y labels
plt.rc('xtick', labelsize = sm)   # fontsize of the tick labels
plt.rc('ytick', labelsize = sm)   # fontsize of the tick labels
plt.rc('legend', fontsize = sm)   # legend fontsize
plt.rc('figure', titlesize = lg)  # fontsize of the figure title
plt.rc('axes', linewidth=2)       # linewidth of plot lines
plt.rcParams['figure.figsize'] = [17,12]
plt.rcParams['figure.dpi'] = 134
plt.rcParams['axes.facecolor'] = 'silver'

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import itertools
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
from dask.distributed import Client
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


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


class The_SARIMA_Model(object):
    def __init__(self, stock):
        self.sss = stock

        import requests
        def get_symbol(symbol):
            url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
            result = requests.get(url).json()
            for x in result['ResultSet']['Result']:
                if x['symbol'] == symbol:
                    return x['name']
        self.company = get_symbol(self.sss)

    def dataHull(self):
        self.start='2011-01-01'
        self.end='2021-01-02'

        self.x_data = yf.download(self.sss, start=self.end)['Adj Close']
        self.x_data.columns = [self.company]

        self.spData = yf.download(self.sss, start=self.start, end=self.end)
        self.dataSP = pd.DataFrame(self.spData['Close'])
        self.dataSP.columns = [self.sss]
        self.dataSP.index = pd.to_datetime(self.dataSP.index)

        self.df_settle = self.spData['Close'].resample('BM').ffill().dropna()
        self.df_rolling = self.df_settle.rolling(12)
        self.df_mean = self.df_rolling.mean()
        self.df_std = self.df_rolling.std()
        st.header(f'Visualization Of The {self.sss} Rolling-12-Day-Mean vs The Stock Price')
        st.subheader('Each intersection indicates either a Buy Or Sell Signal')
        st.write(" * If Stock Price is crossing up above the Rolling-Mean == [BUY]")
        st.write(" * If Stock Price is crossing down below the Rolling-Mean == [SELL]")

        fig, ax = plt.subplots()
        ax.set_xlabel('Dates', fontsize=20, fontweight='bold')
        ax.set_ylabel('Close Prices', fontsize=20, fontweight='bold')
        ax.set_title(f"{self.company} ({self.sss}) - Stock-Price vs Rolling-12-Day-Mean", fontsize=30, fontweight='bold')
        ax.plot(self.df_settle, label='Stock-Price')
        ax.plot(self.df_mean, label='Rolling-12-Day-Mean')
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	        label.set_fontsize(15)
        plt.grid(True, color='k', linestyle='-', linewidth=1, alpha=.3)
        ax.legend(loc='best',prop={"size":16})
        plt.tight_layout()
        st.pyplot(fig)

        st.write(' *'*34)
        st.header("Make Historical Time Series (Stock Price History) 'Stationary'")
        st.write(' *'*34)
        st.header(f'Detrending The TimeSeries History For {self.sss}')
        st.subheader('Detrending')
        st.markdown(' * process of removing trend lines fron a non-stationary data set.')
        st.write(' * involves transformation step that norlalizes large values into smaller ones')
        st.write(' * examples = logarithmic function, square-root function, cube-function')
        st.write(' * further step is to subtract the transformation from the moving average')


    def adf(self):
        self.dataHull()
        self.result = adfuller(self.df_settle)
        st.write('ADF statistic: ',  self.result[0])
        st.write('p-value:', self.result[1])
        self.critical_values = self.result[4]
        for key, value in self.critical_values.items():
            st.write('Critical value (%s): %.3f' % (key, value))
        self.df_log = np.log(self.df_settle)
        self.df_log_ma= self.df_log.rolling(2).mean()
        self.df_detrend = self.df_log - self.df_log_ma
        self.df_detrend.dropna(inplace=True)
      # Mean and standard deviation of detrended data
        self.df_detrend_rolling = self.df_detrend.rolling(12)
        self.df_detrend_ma = self.df_detrend_rolling.mean()
        self.df_detrend_std = self.df_detrend_rolling.std()

        self.result2 = adfuller(self.df_detrend)
        st.write('ADF statistic: ', self.result2[0])
        st.write('p-value: %.5f' % self.result2[1])
        self.critical_values2 = self.result2[4]
        for key, value in self.critical_values2.items():
            st.write('Critical value (%s): %.3f' % (key, value))
        self.df_log_diff = self.df_log.diff(periods=3).dropna()
      # Mean and standard deviation of differenced data
        self.df_diff_rolling = self.df_log_diff.rolling(12)
        self.df_diff_ma = self.df_diff_rolling.mean()
        self.df_diff_std = self.df_diff_rolling.std()

      # Plot the stationary data
        fig, ax = plt.subplots()
        ax.set_xlabel('Dates', fontsize=20, fontweight='bold')
        ax.set_ylabel('Close Prices', fontsize=20, fontweight='bold')
        ax.set_title(f'Removing The Trend In {self.company} ({self.sss}) - By Differencing', fontsize=30, fontweight='bold')
        ax.plot(self.df_log_diff, label='Differenced')
        ax.plot(self.df_diff_ma, label='mean')
        ax.plot(self.df_diff_std, label='std')
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	        label.set_fontsize(15)
        ax.grid(True, color='k', linestyle='-', linewidth=1, alpha=.3)
        ax.legend(loc='best',prop={"size":16})
        plt.tight_layout()
        st.pyplot(fig)
        st.write(' *'*34)        


    def seasonal_decomp(self):
        st.header(f'Configuring Stationary TimeSeries For {self.sss} via Seasonal-Decomposition')
        self.adf()
        self.decompose_result = seasonal_decompose(self.df_log.dropna(), period=12)
        self.df_trend = self.decompose_result.trend
        self.df_season = self.decompose_result.seasonal
        self.df_residual = self.decompose_result.resid      
        self.df_log_diff = self.df_residual.diff().dropna()
      # Mean and standard deviation of differenced data
        self.df_diff_rolling = self.df_log_diff.rolling(12)
        self.df_diff_ma = self.df_diff_rolling.mean()
        self.df_diff_std = self.df_diff_rolling.std()
      # Plot the stationary data
        fig, ax = plt.subplots()
        ax.set_xlabel('Dates', fontsize=20, fontweight='bold')
        ax.set_ylabel('Close Prices', fontsize=20, fontweight='bold')
        plt.title(f'{self.company} ({self.sss}) - Differenced', fontsize=30, fontweight='bold')
        plt.plot(self.df_log_diff, label='Differenced')
        plt.plot(self.df_diff_ma, label='Mean')
        plt.plot(self.df_diff_std, label='Std')
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	        label.set_fontsize(15)
        ax.grid(True, color='k', linestyle='-', linewidth=1, alpha=.3)
        ax.legend(loc='best',prop={"size":16})
        plt.tight_layout()
        st.pyplot(fig)        

        self.result = adfuller(self.df_residual.dropna())
        st.write('ADF statistic:',  self.result[0])
        st.write('p-value: %.5f' % self.result[1])
        self.critical_values = self.result[4]
        for key, value in self.critical_values.items():
            st.write('Critical value (%s): %.3f' % (key, value))


    def arima_grid_search(self, s=12):
        self.seasonal_decomp()
        self.s = s       
        self.p = self.d = self.q = range(2)
        self.param_combinations = list(itertools.product(self.p, self.d, self.q))
        self.lowest_aic, self.pdq, self.pdqs = None, None, None
        self.total_iterations = 0
        for order in self.param_combinations:    
            for (self.p, self.q, self.d) in self.param_combinations:
                self.seasonal_order = (self.p, self.q, self.d, self.s)
                self.total_iterations += 1
                try:
                    self.model = SARIMAX(self.df_settle, order=order, 
                        seasonal_order=self.seasonal_order, 
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        disp=False)
                    self.model_result = self.model.fit(maxiter=200, disp=False)
                    if not self.lowest_aic or self.model_result.aic < self.lowest_aic:
                        self.lowest_aic = self.model_result.aic
                        self.pdq, self.pdqs = order, self.seasonal_order
                except Exception as ex:
                    continue
        st.write('ARIMA{}x{}'.format(self.pdq, self.pdqs))
        st.write('Lowest AIC: ' , self.lowest_aic)
        return self.lowest_aic, self.pdq, self.pdqs


    def fitModel_to_SARIMAX(self):
        self.arima_grid_search()
        self.model = SARIMAX(
            self.df_settle,
            order=self.pdq,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True,
            disp=False)
        self.model_results = self.model.fit(maxiter = 200, disp = False)
        st.write(self.model_results.summary())

        fig, ax = plt.subplots()
        plt.legend(loc='best')
        plt.grid(True, color='k', linestyle='-', linewidth=1, alpha=.3)
        plt.tight_layout()
        st.pyplot(self.model_results.plot_diagnostics(figsize=(15, 12)))
        return self.model_results


    def predict(self):
        self.fitModel_to_SARIMAX()
        self.n = len(self.df_settle.index)
        self.prediction = self.model_results.get_prediction(start=self.n - 12*3, end=self.n + 6)
        self.prediction_ci = self.prediction.conf_int()

        fig, ax = plt.subplots()

        ax = self.df_settle['2018':].plot(label='actual')
        self.prediction_ci.plot(ax=ax, style=['--', '--'], label='Predict')
        self.x_data.plot(style='--',color='k', lw=2, label='Price Since Prediction')
        self.ci_index = self.prediction_ci.index
        self.lower_ci = self.prediction_ci.iloc[:, 0]
        self.upper_ci = self.prediction_ci.iloc[:, 1]
        ax.fill_between(self.ci_index, self.lower_ci, self.upper_ci, color='r', alpha=.09, label='Confidence_Interval_(95%)')
        ax.vlines(['2018-05-01', '2020-01'] ,0, 1, transform=ax.get_xaxis_transform(), colors='k', ls='--', label='Train')
        ax.vlines(['2020-01-01', '2021-01'] ,0, 1, transform=ax.get_xaxis_transform(), colors='r', ls='--', label='Test')
        ax.vlines(['2021-01-01', '2022-06-30'] ,0, 1, transform=ax.get_xaxis_transform(), colors='g', ls='--', label='Prediction')

        ax.set_xlabel('Time (years)', fontsize=20, fontweight='bold')
        ax.set_ylabel('Prices', fontsize=20, fontweight='bold')
        ax.set_title(f'{self.company} ({self.sss}) - SARIMA MODEL', fontsize=30, fontweight='bold')
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	        label.set_fontsize(15)
        fontP = FontProperties()
        fontP.set_size('large')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
        ax.grid(True, color='k', linestyle='-', linewidth=1, alpha=.3)
        plt.tight_layout()
        st.pyplot(fig)
        return


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if __name__ == '__main__':

    stock_ticker = 'BA'

    if stock_ticker:
        The_SARIMA_Model(stock_ticker).predict()


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *