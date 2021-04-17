import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta, date
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
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
plt.rcParams['figure.dpi'] = 150
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


class The_Support_Resistance(object):
    def __init__(self, ticker, period):
        self.ticker = ticker
        self.period = period


        import requests
        def get_symbol(symbol):
            url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
            result = requests.get(url).json()
            for x in result['ResultSet']['Result']:
                if x['symbol'] == symbol:
                    return x['name']
        self.company = get_symbol(self.ticker)


    def setup(self):
        self.df = yf.download(self.ticker, period=self.period, parse_dates=True)
        self.df['Date'] = pd.to_datetime(self.df.index)
        self.df['Date'] = self.df['Date'].apply(mpl_dates.date2num)
        self.df = self.df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]
        return self.df

    def isSupport(self,df,i):
        support = self.df['Low'][i] < self.df['Low'][i-1]  and self.df['Low'][i] < self.df['Low'][i+1] \
        and self.df['Low'][i+1] < self.df['Low'][i+2] and self.df['Low'][i-1] < self.df['Low'][i-2]
        return support

    def isResistance(self,df,i):
        resistance = self.df['High'][i] > self.df['High'][i-1] and self.df['High'][i] > self.df['High'][i+1] \
        and self.df['High'][i+1] > self.df['High'][i+2] and self.df['High'][i-1] > self.df['High'][i-2] 
        return resistance

    def plot_all(self):
        fig, ax = plt.subplots()
        candlestick_ohlc(ax,self.df.values,width=0.6, colorup='green', colordown='red', alpha=0.8)
        date_format = mpl_dates.DateFormatter('%d %b %Y')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        for level in self.levels:
            plt.hlines(level[1],xmin=self.df['Date'][level[0]], xmax=max(self.df['Date']),colors='blue')
            plt.title(f"{self.company} ({self.ticker}) - Support & Resistance Price Levels")
            plt.tight_layout()
            plt.grid(True, linestyle='--')
        fig.show()
        st.pyplot(fig)

        
    def isFarFromLevel(self,l):
        return np.sum([abs(l-x) < self.s  for x in self.levels]) == 0

    def level(self):
        self.setup()
        self.levels = []
        for i in range(2, self.df.shape[0]-2):
            if self.isSupport(self.df,i):
                self.levels.append((i, self.df['Low'][i]))
            elif self.isResistance(self.df, i):
                self.levels.append((i,self.df['High'][i]))
        self.s =  np.mean(self.df['High'] - self.df['Low'])
        self.levels = []
        for i in range(2, self.df.shape[0]-2):
            if self.isSupport(self.df, i):
                l = self.df['Low'][i]
                if self.isFarFromLevel(l):
                    self.levels.append((i,l))
            elif self.isResistance(self.df,i):
                l = self.df['High'][i]
                if self.isFarFromLevel(l):
                    self.levels.append((i,l))
        fd = pd.DataFrame(self.levels)
        fd.columns = ['day', 'price_level']

        new_lst = []
        for i in fd['day']:
            n = int(i)
            enddate = pd.to_datetime(self.df.index[0]) + pd.DateOffset(days=n)
            new_lst.append(enddate)
        fd['date'] = new_lst
        fd.set_index('date', inplace=True)
        st.text('\n')
        st.text(fd)
        self.plot_all()
        plt.show()


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if __name__ == '__main__':
    The_Support_Resistance(ticker='AAPL', period='6mo').level()


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *    