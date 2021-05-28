import pandas as pd
from pandas import Series, DataFrame
from pandas.plotting import scatter_matrix
import numpy as np
import datetime
import math 
import requests
import yfinance as yf
import matplotlib
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import style
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [15,10]
plt.rcParams['figure.dpi'] = 134
from pathlib import Path
from datetime import datetime, timedelta, date
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
from scipy.stats import spearmanr
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
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
    tf.config.experimental.enable_mlir_bridge()
    tf.config.experimental.enable_tensor_float_32_execution(enabled=True)
    tf.config.threading.get_inter_op_parallelism_threads()
    tf.config.threading.set_inter_op_parallelism_threads(0)
else:
    print('Using CPU')

days = 20

class Regression_Model(object):
    def __init__(self, ticker):
        self.ticker = ticker

        def get_symbol_longName(symbol):
            url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
            result = requests.get(url).json()
            for x in result['ResultSet']['Result']:
                if x['symbol'] == symbol:
                    return x['name']
        self.sName = get_symbol_longName(self.ticker)     


    def preprocessing(self):
        start = datetime(2018, 1, 1)
        df = yf.download(self.ticker, start=start, parse_dates=True)
        close_px = df['Adj Close']
        mavg = close_px.rolling(window=100).mean()

        fig, ax = plt.subplots(nrows=1, ncols=1)
        mavg.plot(label='mavg')
        close_px.plot(label='price')
        plt.ylabel(f'{self.sName} Price', fontsize=20, fontweight='bold')
        plt.xlabel('Dates', fontsize=20, fontweight='bold')
        plt.title(f'Moving Average vs {self.sName} ({self.ticker}) - Price', fontsize=30, fontweight='bold')
        plt.xlim(date(2019,1,1))
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	        label.set_fontsize(15)
        ax.grid(True, color='k', linestyle='-', linewidth=1, alpha=.3)
        ax.legend(loc='best',prop={"size":16})        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        retscomp = close_px / close_px.shift(1) - 1
        dfcomp = yf.download(
            [self.ticker, 'AAPL', 'AMZN', 'TSLA', 'GOOGL', 'FB'], start=start)['Adj Close']
        retscomp = dfcomp.pct_change()
        corr = retscomp.corr()

        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.imshow(corr, cmap='hot', interpolation='none')
        plt.colorbar()
        plt.xticks(range(len(corr)), corr.columns)
        plt.yticks(range(len(corr)), corr.columns)
        plt.title(f'HeatMap - Correlation of Returns To: {self.sName} ({self.ticker})')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.scatter(retscomp.mean(), retscomp.std())
        plt.xlabel('Expected returns', fontsize=20, fontweight='bold')
        plt.ylabel('Risk', fontsize=20, fontweight='bold')
        for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
            plt.annotate(
                label, 
                xy = (x, y), xytext = (20, -20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        plt.title(f'Expected Returns Plot Verses {self.sName} ({self.ticker})', fontsize=30, fontweight='bold')
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	        label.set_fontsize(15)
        ax.grid(True, color='k', linestyle='-', linewidth=1, alpha=.3)
        ax.legend(loc='best',prop={"size":16})        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        dfreg = df.loc[:,['Adj Close','Volume']]
        dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
        dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
      # Drop missing value
        dfreg.fillna(value=-99999, inplace=True)
      # We want to separate 1 percent of the data to forecast
        forecast_out = int(math.ceil(0.01 * len(dfreg)))
      # Separating the label here, we want to predict the AdjClose
        forecast_col = 'Adj Close'
        dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
        X = np.array(dfreg.drop(['label'], 1))
      # Scale the X so that everyone can have the same distribution for linear regression
        X = sklearn.preprocessing.scale(X)
      # Finally We want to find Data Series of late X & early X (train) for model generation & eval
        X_lately = X[-forecast_out:]
        X = X[:-forecast_out]
      # Separate label and identify it as y
        y = np.array(dfreg['label'])
        y = y[:-forecast_out]
        X_train, X_test, y_train, y_test = train_test_split(X, y)
      # Linear regression
        clfreg = LinearRegression(n_jobs=-1)
        clfreg.fit(X_train, y_train)
      # Quadratic Regression 2
        clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
        clfpoly2.fit(X_train, y_train)
      # Quadratic Regression 3
        clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
        clfpoly3.fit(X_train, y_train)
      # KNN Regression
        clfknn = KNeighborsRegressor(n_neighbors=2)
        clfknn.fit(X_train, y_train)
      # results
        confidencereg = clfreg.score(X_test, y_test)
        confidencepoly2 = clfpoly2.score(X_test,y_test)
        confidencepoly3 = clfpoly3.score(X_test,y_test)
        confidenceknn = clfknn.score(X_test, y_test)
        st.write('\n * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n')        
        st.write(f' > {self.sName} ({self.ticker}) :\n')
        st.write('The linear regression confidence is ', confidencereg),
        st.write('The quadratic regression 2 confidence is ', confidencepoly2),
        st.write('The quadratic regression 3 confidence is ', confidencepoly3),
        st.write('The knn regression confidence is ', confidenceknn)
        st.write('\n * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \n')

        fd = pd.DataFrame()
        fd['---Regression_Model---'] = ['linear_regression','quadratic_regression_2','quadratic_regression_3','knn']
        fd['Model_Results'] = [confidencereg, confidencepoly2, confidencepoly3, confidenceknn]
        fd.set_index('---Regression_Model---', inplace=True)
        fd.sort_values('Model_Results',ascending=False, inplace=True)
        st.dataframe(fd)
        return dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, fd.index[0]


    def linear_regression(self, dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days):
        forecast_set = clfreg.predict(X_lately)
        dfreg['Forecast'] = np.nan
        dfreg['Forecast']
        last_date = dfreg.iloc[-1].name
        last_unix = last_date
        next_unix = last_unix + timedelta(1)
        for i in forecast_set:
            next_date = next_unix
            next_unix += timedelta(days)
            dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        dfreg['Adj Close'].tail(500).plot()
        dfreg['Forecast'].tail(500).plot()
        plt.title(f'{self.sName} ({self.ticker}) - Linear Regression - 6 Month Forecast', fontsize=30, fontweight='bold')
        plt.xlabel('Date', fontsize=20, fontweight='bold')
        plt.ylabel('Price', fontsize=20, fontweight='bold')
        plt.xlim(date(2020,1,1))
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	        label.set_fontsize(15)
        ax.grid(True, color='k', linestyle='-', linewidth=1, alpha=.3)
        ax.legend(loc='best',prop={"size":16})        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        return

    def quadratic_regression_2(self, dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days):
        forecast_set = clfpoly2.predict(X_lately)
        dfreg['Forecast'] = np.nan
        dfreg['Forecast']
        last_date = dfreg.iloc[-1].name
        last_unix = last_date
        next_unix = last_unix + timedelta(1)
        for i in forecast_set:
            next_date = next_unix
            next_unix += timedelta(days)
            dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        dfreg['Adj Close'].tail(500).plot()
        dfreg['Forecast'].tail(500).plot()
        plt.title(f'{self.sName} ({self.ticker}) - Quadratic (2) Regression - 6 Month Forecast', fontsize=30, fontweight='bold')
        plt.xlabel('Date', fontsize=20, fontweight='bold')
        plt.ylabel('Price', fontsize=20, fontweight='bold')
        plt.xlim(date(2020,1,1))
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	        label.set_fontsize(15)
        ax.grid(True, color='k', linestyle='-', linewidth=1, alpha=.3)
        ax.legend(loc='best',prop={"size":16})        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        return
    
    def quadratic_regression_3(self, dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days):
        forecast_set = clfpoly3.predict(X_lately)
        dfreg['Forecast'] = np.nan
        dfreg['Forecast']
        last_date = dfreg.iloc[-1].name
        last_unix = last_date
        next_unix = last_unix + timedelta(1)
        for i in forecast_set:
            next_date = next_unix
            next_unix += timedelta(days)
            dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        dfreg['Adj Close'].tail(500).plot()
        dfreg['Forecast'].tail(500).plot()
        plt.title(f'{self.sName} ({self.ticker}) - Quadratic (3) Regression - 6 Month Forecast', fontsize=30, fontweight='bold')
        plt.xlabel('Date', fontsize=20, fontweight='bold')
        plt.ylabel('Price', fontsize=20, fontweight='bold')
        plt.xlim(date(2020,1,1))
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	        label.set_fontsize(15)
        ax.grid(True, color='k', linestyle='-', linewidth=1, alpha=.3)
        ax.legend(loc='best',prop={"size":16})        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        return

    def knn(self, dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days):
        forecast_set = clfknn.predict(X_lately)
        dfreg['Forecast'] = np.nan
        dfreg['Forecast']
        last_date = dfreg.iloc[-1].name
        last_unix = last_date
        next_unix = last_unix + timedelta(1)
        for i in forecast_set:
            next_date = next_unix
            next_unix += timedelta(days)
            dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        dfreg['Adj Close'].tail(500).plot()
        dfreg['Forecast'].tail(500).plot()
        plt.title(f'{self.sName} ({self.ticker}) - KNN Regression - 6 Month Forecast', fontsize=30, fontweight='bold')
        plt.xlabel('Date', fontsize=20, fontweight='bold')
        plt.ylabel('Price', fontsize=20, fontweight='bold')
        plt.xlim(date(2020,1,1))
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	        label.set_fontsize(15)
        ax.grid(True, color='k', linestyle='-', linewidth=1, alpha=.3)
        ax.legend(loc='best',prop={"size":16})                
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        return


if __name__ == '__main__':

    stock_ticker = 'TSLA'
    dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, modName = Regression_Model(stock_ticker).preprocessing()

    if modName == 'linear_regression':
        Regression_Model(stock_ticker).linear_regression(dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days)
    if modName == 'quadratic_regression_2':
        Regression_Model(stock_ticker).quadratic_regression_2(dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days)
    if modName == 'quadratic_regression_3':
        Regression_Model(stock_ticker).quadratic_regression_3(dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days)
    if modName == 'knn':
        Regression_Model(stock_ticker).knn(dfreg, X_lately, clfreg, clfpoly2, clfpoly3, clfknn, days)