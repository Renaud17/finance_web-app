#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


import pandas as pd
import numpy as np
import matplotlib
import matplotlib as mpl
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
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib
from mplfinance.original_flavor import candlestick_ohlc
from datetime import datetime
import seaborn as sns
import yfinance as yf
sns.set()
plt.rcParams['figure.figsize'] = [15, 7.5]
plt.rcParams['figure.dpi'] = 134


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


class The_OverBought_OverSold(object):
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


    def generate(self):
        df = yf.download(self.ticker, period=self.period, parse_dates=True)
        df.reset_index(inplace=True)
        df.Date = df.Date.astype('str')
        date = [datetime.strptime(d, '%Y-%m-%d') for d in df['Date']]
        candlesticks = list(
            zip(mdates.date2num(date),df['Open'], df['High'],df['Low'],df['Close'],df['Volume']))


        def removal(signal, repeat):
            copy_signal = np.copy(signal)
            for j in range(repeat):
                for i in range(3, len(signal)):
                    copy_signal[i - 1] = (copy_signal[i - 2] + copy_signal[i]) / 2
            return copy_signal


        def get(original_signal, removed_signal):
            buffer = []
            for i in range(len(removed_signal)):
                buffer.append(original_signal[i] - removed_signal[i])
            return np.array(buffer)


        signal = np.copy(df.Open.values)
        removed_signal = removal(signal, 30)
        noise_open = get(signal, removed_signal)


        signal = np.copy(df.High.values)
        removed_signal = removal(signal, 30)
        noise_high = get(signal, removed_signal)


        signal = np.copy(df.Low.values)
        removed_signal = removal(signal, 30)
        noise_low = get(signal, removed_signal)


        signal = np.copy(df.Close.values)
        removed_signal = removal(signal, 30)
        noise_close = get(signal, removed_signal)


        noise_candlesticks = list(zip(
            mdates.date2num(date),noise_open, noise_high,noise_low,noise_close))
        


        fig = plt.figure()
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        plt.title(f'Analysis of OverBought vs OverSold For {self.company} ({self.ticker})')
        ax1.set_ylabel('Quote ($)', size=20)
        dates = [x[0] for x in candlesticks]
        dates = np.asarray(dates)
        volume = [x[5] for x in candlesticks]
        volume = np.asarray(volume)
        candlestick_ohlc(ax1, candlesticks, width=1, colorup='g', colordown='r')
        pad = 0.25
        yl = ax1.get_ylim()
        ax1.set_ylim(yl[0]-(yl[1]-yl[0])*pad,yl[1])
        plt.grid(True)
        plt.tight_layout()
        
        ax2 = ax1.twinx()
        pos = df['Open'] - df['Close']<0
        neg = df['Open'] - df['Close']>0
        ax2.bar(dates[pos],volume[pos],color='green',width=1,align='center')
        ax2.bar(dates[neg],volume[neg],color='red',width=1,align='center')
        ax2.set_xlim(min(dates),max(dates))
        yticks = ax2.get_yticks()
        ax2.set_yticks(yticks[::3])
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel('Volume', size=20)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        ax2.set_ylabel('Quote ($)', size=20)
        candlestick_ohlc(ax2, noise_candlesticks, width=1, colorup='g', colordown='r')
        ax2.plot(
            dates, 
            [np.percentile(noise_close, 95)] * len(noise_candlesticks), 
            color = (1.0, 0.792156862745098, 0.8, 1.0),
            linewidth=5.0, label = 'overbought line'
            )
        ax2.plot(
            dates, 
            [np.percentile(noise_close, 10)] * len(noise_candlesticks), 
            color = (0.6627450980392157, 1.0, 0.6392156862745098, 1.0),
            linewidth=5.0, label = 'oversold line'
            )
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(10))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(fig)


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if __name__ == '__main__':
    The_OverBought_OverSold(ticker='IKE.NZ', period='1y').generate()


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *    