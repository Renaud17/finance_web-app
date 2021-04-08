#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
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
plt.style.use('ggplot')
# [
#   'Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 
#   'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 
#   'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 
#   'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10'
# ]
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


class Web_prophet_kyle(object):
    def __init__(self, stock_ticker=None):   
        pass    

    def make_forecast1(self, stonk, per, hist='1y'):
        """forecast the given ticker (stock) period days into the future (from today)
        ---------inputs----------
        > ticker ->> ticker of stock to forecast
        > periods->> number of days into the future to forecast (from today's date)
        > hist   ->> amount of historical data to use [default=max] -> options(1d,5d,1mo,3mo,6mo,1y,2y,5y,10y}"""
        self.stonk = stonk
        self.per = per
      # pull historical data from yahoo finance
        stock_data = yf.Ticker(self.stonk)
        df = stock_data.history(hist, auto_adjust=True)
        df.reset_index(inplace=True)
        df.fillna(0.0, inplace=True)
        df = df[["Date","Close"]] # select Date and Price
        df = df.rename(columns = {"Date":"ds","Close":"y"})

        import requests
        def get_symbol(symbol):
            url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
            result = requests.get(url).json()
            for x in result['ResultSet']['Result']:
                if x['symbol'] == symbol:
                    return x['name']
        self.company = get_symbol(self.stonk) 


      # create a Prophet model from that data
        m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.1, seasonality_prior_scale=10)
        m.fit(df)
        future = m.make_future_dataframe(self.per, freq='D')
        forecast = m.predict(future)
        forecast = forecast[['ds', 'trend', 'yhat_lower', 'yhat_upper', 'yhat']]
      # create plot
        fig1 = m.plot(forecast,ax=None,uncertainty=True,plot_cap=True,xlabel='Date',ylabel='Stock Price')
        add_changepoints_to_plot(fig1.gca(), m, forecast)
        plt.title(f"Prophet Model ChangePoints - {self.company} ({self.stonk}) - {self.per} Day Forecast")
        plt.legend(['actual', 'prediction', 'changePoint_line'], loc='best')
        st.pyplot(fig1)


      # create a Prophet model from that data
        m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.1, seasonality_prior_scale=10)
        forecast_2 = m.fit(df).predict(future)
      # create plot
        fig2 = m.plot(
        forecast_2,ax=None,uncertainty=True,plot_cap=True,xlabel='Date',ylabel='Stock Price'
        )
        plt.title(f"Prophet Model Prediction - {self.company} ({self.stonk}) - {self.per} Day Forecast")
        plt.legend(['actual', 'prediction', 'confidence_interval'],loc='best')
        st.pyplot(fig2)
      # create datatable
        forecast_df = forecast[['ds', 'trend', 'yhat_lower', 'yhat_upper', 'yhat']]
        forecast_df.columns = [['date','trend','lower_confidence_interval','upper_confidence_interval','prediction']]
        st.table(forecast_df.tail())


      # create a Prophet model from that data
        m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.1, seasonality_prior_scale=10)
        m.fit(df)
        future = m.make_future_dataframe(periods=self.per)
        forecast = m.predict(future)
      # create plot
        fig5 = m.plot_components(forecast)
        st.header(f'Components Model For {self.company} ({self.stonk}) - Prophet Model')
        st.pyplot(fig5)

        return


#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


    def make_forecast2(self, stonk, per, hist='10y'): 
        """forecast the given ticker (stock) period days into the future (from today)
        ---------inputs----------
        > ticker ->> ticker of stock to forecast
        > periods->> number of days into the future to forecast (from today's date)
        > hist   ->> amount of historical data to use [default=max] -> options(1d,5d,1mo,3mo,6mo,1y,2y,5y,10y}"""
        self.stonk2 = stonk
        self.per2 = per
      # pull historical data from yahoo finance
        stock_data = yf.Ticker(self.stonk2)
        df = stock_data.history(hist, auto_adjust=True)
        df.reset_index(inplace=True)
        df.fillna(0.0, inplace=True)
        df = df[["Date","Close"]] # select Date and Price
        df = df.rename(columns = {"Date":"ds","Close":"y"})
        
        m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.1, seasonality_prior_scale=10)
        m.fit(df)

        import requests
        def get_symbol(symbol):
            url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
            result = requests.get(url).json()
            for x in result['ResultSet']['Result']:
                if x['symbol'] == symbol:
                    return x['name']
        self.company2 = get_symbol(self.stonk2)        


      # create Cross Validation Model
        df_cv = cross_validation(m, initial='360 days', period='90 days', horizon = '180 days')
        df_p = performance_metrics(df_cv)
        
        cutoffs = pd.to_datetime(['2019-02-15', '2019-08-15', '2020-02-15'])
        df_cv2 = cross_validation(m, cutoffs=cutoffs, horizon='180 days')

      # create plot
        fig = plot_cross_validation_metric(df_cv, metric='mape',figsize=(17,8))
        plt.title(f'Cross Validation Model For {self.company2} ({self.stonk2}) - Prophet Model')
        plt.legend(['MeanAvgError(line)/perdictionCount', 'Avg-MeanError(dots)/perdictionCount'], loc='best')
        st.pyplot(fig)


      # connect to the cluster
        client = Client()
        df_cv = cross_validation(m, initial='360 days', period='90 days', horizon='180 days', parallel="dask")
        param_grid={
            'changepoint_prior_scale':[.001,.01,.1,.5],
            'seasonality_prior_scale':[.01,.1,1.0,10.0],
        }
      # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v))
                    for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here


      # Use cross validation to evaluate all parameters
        for params in all_params:
            m = Prophet(**params).fit(df)  # Fit model with given params
            df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days', parallel="processes")
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p['rmse'].values[0])


      # Find the ALL parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        st.subheader('All Tuning Results:')
        st.dataframe(tuning_results)

      # Find the best parameters
        best_params = all_params[np.argmin(rmses)]
        st.subheader('Best Paramaters:')
        st.text(best_params)
        return


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *