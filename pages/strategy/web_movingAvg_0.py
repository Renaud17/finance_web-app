#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


import pandas as pd
import matplotlib
import matplotlib as mpl
matplotlib.use('Agg')
from matplotlib import style
from matplotlib import pyplot as plt
# plt.style.use('seaborn-talk')
# plt.style.use('seaborn-poster')
# plt.style.use('_classic_test_patch')
# plt.style.use('fast')
# plt.style.use('fivethirtyeight')
# plt.style.use('seaborn-dark-palette')
# plt.style.use('seaborn-colorblind')
# plt.style.use('seaborn-deep')
# plt.style.use('seaborn-muted')
# plt.style.use('seaborn-notebook')
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
import yfinance as yf
from datetime import datetime
import numpy as np
plt.rcParams['figure.dpi'] = 134
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


def ST_Trading_signals(ticker):
    # start_date = datetime(2020, 1, 1)
    # end_date = datetime.now()
    # df = yf.download(ticker, start=start_date, end=end_date)
    df = yf.download(ticker, period='1y')


    import requests
    def get_symbol(symbol):
        url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
        result = requests.get(url).json()
        for x in result['ResultSet']['Result']:
            if x['symbol'] == symbol:
                return x['name']
    company = get_symbol(ticker)


    df_daily_close = df[['Adj Close']]
    df_daily_pct_c = df_daily_close.pct_change()
    df_daily_pct_c.fillna(0, inplace=True)
    df_daily_pct_c = df_daily_close / df_daily_close.shift(1) - 1
    df['Daily_S_RoR'] = df_daily_pct_c['Adj Close']

  # LOG Rate Of Return
    df_daily_log_returns = np.log(df_daily_close.pct_change()+1)
    df['Daily_Log'] = df_daily_log_returns['Adj Close']

  # Total Return
    df_cum_daily_return = (1 + df_daily_pct_c).cumprod()
    df['Total_RoR'] = df_cum_daily_return['Adj Close']
    df.rename(columns={'Adj Close': ticker}, inplace=True)
    short_window = 2
    long_window = 20

  # Initialize the `signals` DataFrame with the `signal` column
    signals = pd.DataFrame(index=df.index)
  # Create short simple moving average over the short window
    signals['signal'] = 0.0                                     
    signals['short_mavg'] = df['Close'].rolling(
        window=short_window,min_periods=1,center=False).mean()
    signals['long_mavg'] = df['Close'].rolling(
        window=long_window, min_periods=1, center=False).mean()
    signals['signal'][short_window:] = np.where(
        signals['short_mavg'][short_window:]> signals['long_mavg'][short_window:], 1.0, 0.0
        )
    signals['positions'] = signals['signal'].diff()              

    fig = plt.figure(figsize=(15, 11))                                    
    ax1 = fig.add_subplot(111,  ylabel='Price in $')                       
    df['Close'].plot(ax=ax1, lw=2.)                                      
    signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)              
    ax1.plot(
        signals.loc[signals.positions == 1.0].index, 
        signals.short_mavg[signals.positions == 1.0],'^', markersize=15, color='k'
        )
    ax1.plot(
        signals.loc[signals.positions == -1.0].index, 
        signals.short_mavg[signals.positions == -1.0],'v', markersize=15, color='r'
        )
    ax1.set_title(f"{company} ({ticker}) - Moving Average Trade Signals SHORT-TERM")
    plt.legend(loc='best')
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    

if __name__ == '__main__':
    ticker = 'TSLA'
    ST_Trading_signals(ticker)


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *    