import warnings
warnings.filterwarnings('ignore')
import yahoo_fin.stock_info as si
from pathlib import Path
import numpy as np 
import pandas as pd
import matplotlib
import matplotlib as mpl
from itertools import product
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
import datetime
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')
import pandas_datareader.data as web
import streamlit as st
import yfinance as yf
import requests


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

def clean(listA):
    lst = list(set(listA))
    lst.sort()
    return lst

saveTickers = Path('/home/gordon/gdp/code/portfolio/Forecasting_For_Friends/tickers/')
my_positions = pd.read_pickle(saveTickers / f'chuck_merged_ticker_lst.pkl')
watch_lst0 = pd.read_pickle(saveTickers / f'watch_merged_ticker_lst.pkl')
watch_lst_bulk = list(set(my_positions + watch_lst0))

dow = pd.read_pickle(saveTickers / f'dow_ticker_lst.pkl')
sp100 = pd.read_pickle(saveTickers / f'sp100_ticker_lst.pkl')
sp500 = pd.read_pickle(saveTickers / f'sp500_ticker_lst.pkl')
indices_main = ['^OEX','^MID','^GSPC','^DJI','^NYA','^RUT','^W5000']
day_gainers = list(si.get_day_gainers()['Symbol'])
day_losers = list(si.get_day_losers()['Symbol'])
day_most_active = list(si.get_day_most_active()['Symbol'])
undervalued_large_caps = list(si.get_undervalued_large_caps()['Symbol'])
fool_composite = [
    'LMND','ZM','TTD','PINS','TEAM','SAM','DIS','ASML','ECL','NYT',
    'LRCX','NTDOY','PYPL','AMZN','ABNB','ATVI','ZM','SKLZ','SHOP', 'STAA',
    'LULU','WING', 'ETSY','BL','RDFN','LOGI','EQIX','U','RGEN','CHGG',
    'PINS','FUBO','W','MRNA','AXON','SNBR','TDOC','GDRX','PLNT','PLNT',
    'GDRX','RDFN','PINS','LULU','AVAV','FSLY','AXON','BL','ZEN','DDOG',
    'NEE','CRM','TEAM','ZG','Z','TWLO','RMD','STAA','WING','ETSY',
    'LOGI','EQIX','U','RGEN','CHGG','FUBO','LO','MRNA','SNBR','TDOC',
    'AAPL','ASML','BLDP','BA','CRLBF','GWPH','NVDA','PLTR','SNOW','SIVB',
    'TSLA','HRVSF','ARKG','ARKK','GBTC','ECL','NNOX','OKTA','CRM','SOHU',
    'FVRR','LTCN','OROCF','ETCG','APHA','BILI','CLLS','COUP','CUE','NYT',
    'RIOT','SE','SQ','TECK'    
]
oxford_composite = [
  'BZH','CROX','HA','HAS','PFGC','POOL','GDOT','HUYA','GRUB','FSLR','SPWR',
  'GS','BYND','PFGC','VRA','NLOK','NET','BYND','YETI','CURLF','ALB','WMT',
  'DG','BCO','NFE','UBER','RUN','BABA','FAST','CRLBF','LUN-T','TRIP','FCEL',
  'ALB','BABA','FAST','FCEL','LUN-T','YETI','BCO','DG','DLR','GRMN','WMT',
  'LSCC','MU','NVDA','TRIP','UBER','CRLBF','CURLF','IIPR','CVS','XLE','IBM',
  'ORCL','VZ','DEM','WIX','ZEN','VWEHX, VSMAX','VIPSX','VFSTX','VEMAX',
  'VTSAX','VPADX','VGSLX','VEUSX','GDX','GKOS','MRVL','NEO','NVCR','PFPT',
  'TDF','PSHZF','MKL','IEP','EQR','EMF','BRK-B','AIG','F','PFE','V','WMT',
  'JLL','MCK','RDS-B','BUD','DAL','RIOT','MRNA','AMKBY','BRPHF','ZEST',
  'WIX','PSHZF','ZEN','IBM','VZ','XLE','ORCL','NEO','LBTYA','DRNA','EA',
  'EGHT','BZUN','TTWO','EBAY','FTNT','SAIL','BABA','SAM','COLM','DPZ',
  'EXPE','NUVA','EA','HSY','HAS','NFLX','SIX'
]
index_ticker_lists_A = [
  dow, sp100, sp500, indices_main, watch_lst_bulk, fool_composite, oxford_composite,
  day_gainers, day_losers, day_most_active, undervalued_large_caps
]
index_ticker_lists_B = [
  'dow', 'sp100', 'sp500', 'indices_main','watch_lst_bulk', 'fool_composite', 'oxford_composite',
  'day_gainers', 'day_losers', 'day_most_active', 'undervalued_large_caps'
]
for r in range(len(index_ticker_lists_A)):
  index_ticker_lists_A[r] = clean(index_ticker_lists_A[r])


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


class The_Strategy_2(object):
    def __init__(self, tic, sName):
        self.tic = tic
        self.sName = sName

    def grab_data(self):
        ticker = yf.Ticker(self.tic)
        self.raw0 = ticker.history(period='2y')
        self.raw = pd.DataFrame(self.raw0)
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

        sma1 = range(2, 81, 2)
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
        return S, L, results[:1], self.raw0[self.sName]


def movAvgCrossStrategy(stock_symbol, longName, short_window, long_window, moving_avg, raw, display_table=True):
    '''
    The function takes the stock symbol, time-duration of analysis, 
    look-back periods and the moving-average type(SMA or EMA) as input 
    and returns the respective MA Crossover chart along with the buy/sell signals for the given period.
    - stock_symbol - (str)stock ticker as on Yahoo finance. Eg: 'ULTRACEMCO.NS' 
    - short_window - (int)lookback period for short-term moving average. Eg: 5, 10, 20 
    - long_window - (int)lookback period for long-term moving average. Eg: 50, 100, 200
    - moving_avg - (str)the type of moving average to use ('SMA' or 'EMA')
    - display_table - (bool)whether to display the date and price table at buy/sell positions(True/False)
    '''
  # import the closing price data of the stock for the aforementioned period of time in Pandas dataframe
    stock_df = pd.DataFrame(raw) # convert Series object to dataframe 
    stock_df.columns = {'Close Price'} # assign new colun name
    stock_df.dropna(axis = 0, inplace = True) # remove any null rows

  # column names for long and short moving average columns
    short_window_col = str(short_window) + '_' + moving_avg
    long_window_col = str(long_window) + '_' + moving_avg  

    if moving_avg == 'SMA':
      # Create a short& long simple moving average column
        stock_df[short_window_col] = stock_df['Close Price'].rolling(window = short_window, min_periods = 1).mean()
        stock_df[long_window_col] = stock_df['Close Price'].rolling(window = long_window, min_periods = 1).mean()
    elif moving_avg == 'EMA':
      # Create short & long exponential moving average column
        stock_df[short_window_col] = stock_df['Close Price'].ewm(span = short_window, adjust = False).mean()
        stock_df[long_window_col] = stock_df['Close Price'].ewm(span = long_window, adjust = False).mean()

  # create new column 'Signal' so if faster moving average is greater than slower moving average = set Signal as 1 else 0.
    stock_df['Signal'] = 0.0  
    stock_df['Signal'] = np.where(stock_df[short_window_col] > stock_df[long_window_col], 1.0, 0.0) 
  # create a new column 'Position' which is a day-to-day difference of the 'Signal' column. 
    stock_df['Position'] = stock_df['Signal'].diff()

    df_pos = stock_df[(stock_df['Position'] == 1) | (stock_df['Position'] == -1)]
    df_pos['Position'] = df_pos['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
    
    if df_pos['Position'][-1] == 'Buy':
        buy_lst.append(stock_symbol)
    if df_pos['Position'][-1] == 'Sell':
        sell_lst.append(stock_symbol)


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if __name__ == '__main__':

  for r in range(len(index_ticker_lists_B)):
    st.title(index_ticker_lists_B[r])
    buy_lst = []
    sell_lst = []
    stock_ticker = index_ticker_lists_A[r]
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    for ss in stock_ticker:
        def get_symbol_longName(symbol):
            url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
            result = requests.get(url).json()
            for x in result['ResultSet']['Result']:
                if x['symbol'] == symbol:
                    return x['name']
        company_longName = get_symbol_longName(ss)    

        S, L, res, raw1  = The_Strategy_2(ss, company_longName).grab_data()
        cross=movAvgCrossStrategy(ss, company_longName, S, L, 'SMA', raw1)        

    st.header('BUY LIST')
    st.subheader(buy_lst)
    
    st.header('SELL LIST')
    st.subheader(sell_lst)


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *