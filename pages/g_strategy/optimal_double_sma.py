import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression  
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import datetime as dt
from pylab import mpl, plt
import yfinance as yf
from itertools import product
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('seaborn') 
sm, med, lg = 10, 15, 25
plt.rc('font', size = sm)                                           # controls default text sizes
plt.rc('axes', labelsize = med)                                     # fontsize of the x & y labels
plt.rc('axes', titlesize = med)                                     # fontsize of the axes title
plt.rc('xtick', labelsize = sm)                                     # fontsize of the tick labels
plt.rc('ytick', labelsize = sm)                                     # fontsize of the tick labels
plt.rc('legend', fontsize = med)                                     # legend fontsize
plt.rc('figure', titlesize = lg)                                    # fontsize of the figure title
plt.rc('axes', linewidth=2)                                         # linewidth of plot lines
plt.rcParams['legend.fontsize'] = 'medium'
legend_properties = {'weight':'bold'}
plt.rcParams['figure.figsize'] = [13, 6.5]
plt.rcParams['figure.dpi'] = 134
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.borderpad"] = 0.9
plt.rcParams["legend.framealpha"] = 0.1
plt.rcParams["axes.facecolor"] = 'white'
plt.rcParams["axes.edgecolor"] = 'black'
plt.rcParams["legend.loc"] = 'upper left'
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fancybox"] = True
pd.set_option('display.max_rows', 25)
import requests
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
    
    
class The_Strategy_2(object):
    def __init__(self, tic, sName):
        self.tic = tic
        self.sName = sName

    def grab_data(self):
        ticker = yf.Ticker(self.tic)
        self.raw = ticker.history(period='max')
        self.raw.columns = ['Open', 'High', 'Low', self.sName, 'Volume', 'Dividends', 'Stock Splits']
        SMA1 = 2
        SMA2 = 5
        data1= pd.DataFrame(self.raw[self.sName])
        data1.columns = [self.sName]
        data1['SMA1'] = data1[self.sName].rolling(SMA1).mean()
        data1['SMA2'] = data1[self.sName].rolling(SMA2).mean()
        data1['Position'] = np.where(data1['SMA1'] > data1['SMA2'], 1, -1)
        data1['Returns'] = np.log(data1[self.sName] / data1[self.sName].shift(1))
        data1['Strategy'] = data1['Position'].shift(1) * data1['Returns']
        data1.round(4).tail()
        data1.dropna(inplace=True)
        np.exp(data1[['Returns', 'Strategy']].sum())
        np.exp(data1[['Returns', 'Strategy']].std() * 252**0.5)

        sma1 = range(2, 76, 2)
        sma2 = range(5, 202, 5)
        results = pd.DataFrame()
        for SMA1, SMA2 in product(sma1, sma2):
            data1 = pd.DataFrame(self.raw[self.sName])
            data1.dropna(inplace=True)
            data1['Returns'] = np.log(data1[self.sName] / data1[self.sName].shift(1))
            data1['SMA1'] = data1[self.sName].rolling(SMA1).mean()             
            data1['SMA2'] = data1[self.sName].rolling(SMA2).mean()             
            data1.dropna(inplace=True)             
            data1['Position'] = np.where(data1['SMA1'] > data1['SMA2'], 1, -1)             
            data1['Strategy'] = data1['Position'].shift(1) * data1['Returns']             
            data1.dropna(inplace=True)             
            perf = np.exp(data1[['Returns', 'Strategy']].sum())             
            results = results.append(pd.DataFrame(
                {'SMA1': SMA1, 'SMA2': SMA2,                          
                'MARKET(%)': perf['Returns'],                          
                'STRATEGY(%)': perf['Strategy'],                          
                'OUT': (perf['Strategy'] - perf['Returns'])
                }, index=[0]), ignore_index=True
            )
        results = results.loc[results['SMA1'] < results['SMA2']]
        results = results.sort_values('OUT', ascending=False).reset_index(drop=True).head(10)   
        S, L, mkt, strat, out = results['SMA1'][0], results['SMA2'][0], results['MARKET(%)'][0], results['STRATEGY(%)'][0], results['OUT'][0]

        st.title("Double Moving Average Strategy")
        st.header(f"{self.sName} ({self.tic})")
        st.subheader(f"\n({self.tic}) {self.sName} - Best Short/Long Intervals = {S} & {L}\n")
        st.dataframe(results[:3])        

        return S, L

#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
#  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

if __name__ =='__main__':

    ticker = 'LAZR'

    def get_symbol_longName(symbol):
        url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
        result = requests.get(url).json()
        for x in result['ResultSet']['Result']:
            if x['symbol'] == symbol:
                return x['name']
    company_longName = get_symbol_longName(ticker)

    Short, Long  = The_Strategy_2(ticker, company_longName).grab_data()
    st.write(f"\nBest Short/Long Intervals = {Short} & {Long}\n")