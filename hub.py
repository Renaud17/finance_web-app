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

from pages import forecast as f1
from pages import strategy as f2
from pages import portfolio as f3


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


def clean(listA):
    lst = list(set(listA))
    lst.sort()
    return lst

saveTickers = Path('tickers/')
dow = pd.read_pickle(saveTickers / f'dow_ticker_lst.pkl')
sp100 = pd.read_pickle(saveTickers / f'sp100_ticker_lst.pkl')
sp500 = pd.read_pickle(saveTickers / f'sp500_ticker_lst.pkl')
indices_main = ['^OEX','^MID','^GSPC','^DJI','^NYA','^RUT','^W5000']
index_names = ['SP100','SP400','SP500','DOW','NYSE','Russ2k','Wilshire5k']
combined_index_main_names = [list(x) for x in zip(indices_main, index_names)]
my_positions = pd.read_pickle(saveTickers / f'chuck_merged_ticker_lst.pkl')
watch_lst0 = pd.read_pickle(saveTickers / f'watch_merged_ticker_lst.pkl')
watch_lst_bulk = my_positions + watch_lst0

fool_stock_advisor = [
    'LMND','ZM','TTD','PINS','TEAM','SAM','DIS','ASML','ECL','NYT','LRCX',
    'NTDOY','PYPL','AMZN','ABNB','ATVI','ZM','SKLZ','SHOP', 'STAA','LULU','WING',
    'ETSY','BL','RDFN','LOGI','EQIX','U','RGEN','CHGG','PINS','FUBO','W','MRNA','AXON',
    'SNBR','TDOC','GDRX','PLNT'
]
fool_rule_breakers = [
  'PLNT','GDRX','RDFN','PINS','LULU','AVAV','FSLY','AXON','BL','ZEN','DDOG','NEE','CRM','TEAM','ZG','Z','TWLO',
  'RMD','STAA','WING','ETSY','LOGI','EQIX','U','RGEN','CHGG','FUBO','LO','MRNA','SNBR','TDOC'
]
oxford_dynamicFortunes = [
  'BZH','CROX','HA','HAS','PFGC','POOL','GDOT','HUYA','GRUB','FSLR','SPWR','GS','BYND','PFGC','VRA'
]
oxford_strategicTrends = [
  'NLOK','NET','BYND','YETI','CURLF','ALB','WMT','DG','BCO','NFE','UBER','RUN','BABA','FAST','CRLBF','LUN-T','TRIP','FCEL'
]
oxford_ST_fortuneHunters = [
  'ALB','BABA','FAST','FCEL','LUN-T','YETI'
]
oxford_ST_foundation = [
  'BCO','DG','DLR','GRMN','WMT'
]
oxford_ST_trailblazer = [
  'LSCC','MU','NVDA','TRIP','UBER'
]
oxford_ST_reefer = [
  'CRLBF','CURLF','IIPR'
]
oxford_trading = [
  'CVS','XLE','IBM','ORCL','VZ','DEM','WIX','ZEN'
]
oxford_goneFishin = [
  'VWEHX, VSMAX','VIPSX','VFSTX','VEMAX','VTSAX','VPADX','VGSLX','VEUSX','GDX'
]
oxford_baggers = [
  'GKOS','MRVL','NEO','NVCR','PFPT'
]
oxford_allstars = [
  'TDF','PSHZF','MKL','IEP','EQR','EMF','BRK-B'
]
blockchain = [
  'AIG','F','PFE','V','WMT','JLL','MCK','RDS-B','BUD','DAL','RIOT','MRNA','AMKBY','BRPHF','ZEST'
]
oxford_comminique = [
  'WIX','PSHZF','ZEN','IBM','VZ','XLE','ORCL','NEO','LBTYA','DRNA'
]
oxford_CASH_list = [
  'EA','EGHT','BZUN','TTWO','EBAY','FTNT','SAIL','BABA',
  'SAM','COLM','DPZ','EXPE','NUVA','EA','HSY','HAS','NFLX','SIX'
]
my_new_kicks = [
  'AAPL','ASML','BLDP','BA','CRLBF','GWPH','NVDA','PLTR','SNOW','SIVB','TSLA','HRVSF','ARKG','ARKK','GBTC','ECL','NNOX','OKTA','CRM','SOHU',
  'FVRR','LTCN','OROCF','ETCG','APHA','BILI','CLLS','COUP','CUE','NYT','RIOT','SE','SQ','TECK'
]

snp500 = pd.read_csv("files/SP500.csv")
symbols = snp500['Symbol'].sort_values().tolist()

fool_stock_advisor = clean(fool_stock_advisor)
fool_rule_breakers = clean(fool_rule_breakers)
oxford_dynamicFortunes = clean(oxford_dynamicFortunes)
oxford_strategicTrends = clean(oxford_strategicTrends)
oxford_ST_fortuneHunters = clean(oxford_ST_fortuneHunters)
oxford_ST_foundation = clean(oxford_ST_foundation)
oxford_ST_trailblazer = clean(oxford_ST_trailblazer)
oxford_ST_reefer = clean(oxford_ST_reefer)
oxford_trading = clean(oxford_trading)
oxford_goneFishin = clean(oxford_goneFishin)
oxford_baggers = clean(oxford_baggers)
oxford_allstars = clean(oxford_allstars)
blockchain = clean(blockchain)
oxford_comminique = clean(oxford_comminique)
oxford_CASH_list = clean(oxford_CASH_list)
my_new_kicks = clean(my_new_kicks)

index_ticker_lists_A = [
  fool_stock_advisor, fool_rule_breakers, oxford_dynamicFortunes, oxford_strategicTrends, oxford_ST_fortuneHunters, 
  oxford_ST_foundation, oxford_ST_trailblazer, oxford_ST_reefer, oxford_trading, oxford_goneFishin, oxford_baggers, 
  oxford_allstars, blockchain, oxford_comminique, oxford_CASH_list, my_new_kicks, watch_lst_bulk,
  dow, sp100, sp500
]
index_ticker_lists_B = [
  'fool_stock_advisor', 'fool_rule_breakers', 'oxford_dynamicFortunes', 'oxford_strategicTrends', 'oxford_ST_fortuneHunters', 
  'oxford_ST_foundation', 'oxford_ST_trailblazer', 'oxford_ST_reefer', 'oxford_trading', 'oxford_goneFishin', 'oxford_baggers', 
  'oxford_allstars', 'blockchain', 'oxford_comminique', 'oxford_CASH_list', 'my_new_kicks', 'watch_lst_bulk',
  'dow', 'sp100', 'sp500'
]


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


st.sidebar.subheader('> Step #1')
systemStage = st.sidebar.selectbox('Select Analysis Category:',
  [
    '-Select-Stage-','Fundamental-Analysis', 'Technical-Analysis', 'Portfolio', 'Forecasting','Strategy'
  ]
)
st.sidebar.write(' *'*25)
if(systemStage=='-Select-Stage-'):
  st.title("Welcome To The 'Fin-Web-App'")
  st.write("* A web application designed specifically with the goal to bring the benefit of complex machine learning models & techniques to the average individual.\
    This program provides a wide range of analytical data analysis tools to uncover what information resides within the underlying data to more acurately interperate the \
      market.  In doing so and emplorinng the prinicipal concepts at the heart of data science this application attempts to narrow the field of potential investments\
         from the entire stock market down to a targeted selection of securities with the aim to outperform the broader market index.")
  st.header("This web application is broken into several Stages:")
  st.subheader('1) Fundamental Analysis')
  st.subheader('2) Technical Analysis')
  st.subheader('3) Portfolio Theory & Construction')
  st.subheader('4) Forecasting Techniques')
  st.subheader('5) Trading Strategies & Backtesting')

  st.title("To begin using the models within this web-app, locate the '>' in the upper LEFT hand corner of the screen")
  st.write('')
  st.write('')
  st.subheader("All Interaction & Inputs will work through the side-pannel that will pop up when you click on the '>'")
  st.write('')
  st.write("* Follow the Steps down the side pannel for each model and it will indicate you to hit a 'RUN' button at the bottom")
  st.write("* When yor are ready to Access, Configure, & Run the models in each stage, Select A Stage in Step #1 on the Side Bar to the left.")


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if(systemStage == 'Fundamental-Analysis'):
  st.title('Fundamental Analysis Home Page')
  st.write(' *'*25)
  st.header('General Analysis Notes')
  st.write("https://www.investopedia.com/terms/f/fundamentalanalysis.asp")
  st.write(' *'*25)
  st.subheader('Definition:')
  st.write('* Fundamental analysis is a method of evaluating the intrinsic value of an asset and analyzing the factors that could influence its \
    price in the future. This form of analysis is based on external events and influences, as well as financial statements and industry trends.')
  st.write("* Fundamental analysts are concerned with the difference between a stock's value, and the price at which it is trading.")
  st.write(' *'*25)
  st.subheader('The 6 Segments to perform fundamental analysis on stocks')
  st.write("1) Use the financial ratios for initial screening")
  st.write("2) Understand the company")
  st.write("3) Study the financial reports of the company")
  st.write("4) Check the debt and red signs")
  st.write("5) Find the company's competitors")
  st.write("6) Analyse the future prospects.")
  st.write(' *'*25)
  st.subheader("KEY TAKEAWAYS")
  st.write("* Fundamental analysis is a method of determining a stock's real or 'fair market' value.")
  st.write("* Fundamental analysts search for stocks that are currently trading at prices that are higher or lower than their real value.")
  st.write("* If the fair market value is higher than the market price, the stock is deemed to be undervalued and a buy recommendation is given.")
  st.write("* In contrast, technical analysts ignore the fundamentals in favor of studying the historical price trends of the stock.")
  st.write(' *'*25)
  st.header('Model Results Below:')

  st.sidebar.subheader('> Step #2')
  ticker = st.sidebar.text_input('Enter Stock Ticker IN ALL CAPS')
  if ticker:
    st.sidebar.subheader('Ticker Input = Good')
    st.sidebar.write(' *'*25)
    import requests
    def get_symbol(symbol):
        url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
        result = requests.get(url).json()
        for x in result['ResultSet']['Result']:
            if x['symbol'] == symbol:
                return x['name']
    fundamental_company = get_symbol(ticker)

    st.sidebar.subheader('> Step #3')
    st.sidebar.markdown(f"Hit 'Run' For Fundamental Analysis On:\n {fundamental_company} ({ticker})")
    run_button = st.sidebar.button("RUN")
    if run_button:
      stock = yf.Ticker(ticker)
      info = stock.info
      st.title('Company Profile')
      st.subheader(info['longName']) 
      st.markdown('** Sector **: ' + info['sector'])
      st.markdown('** Industry **: ' + info['industry'])
      st.markdown('** Phone **: ' + info['phone'])
      st.markdown('** Address **: ' + info['address1'] + ', ' + info['city'] + ', ' + info['zip'] + ', '  +  info['country'])
      st.markdown('** Website **: ' + info['website'])
      st.markdown('** Business Summary **')
      st.info(info['longBusinessSummary'])
      fundInfo = {
              'Enterprise Value (USD)': info['enterpriseValue'],
              'Enterprise To Revenue Ratio': info['enterpriseToRevenue'],
              'Enterprise To Ebitda Ratio': info['enterpriseToEbitda'],
              'Net Income (USD)': info['netIncomeToCommon'],
              'Profit Margin Ratio': info['profitMargins'],
              'Forward PE Ratio': info['forwardPE'],
              'PEG Ratio': info['pegRatio'],
              'Price to Book Ratio': info['priceToBook'],
              'Forward EPS (USD)': info['forwardEps'],
              'Beta ': info['beta'],
              'Book Value (USD)': info['bookValue'],
              'Dividend Rate (%)': info['dividendRate'], 
              'Dividend Yield (%)': info['dividendYield'],
              'Five year Avg Dividend Yield (%)': info['fiveYearAvgDividendYield'],
              'Payout Ratio': info['payoutRatio']
          }
      fundDF = pd.DataFrame.from_dict(fundInfo, orient='index')
      fundDF = fundDF.rename(columns={0: 'Value'})
      st.subheader('Fundamental Info') 
      st.table(fundDF)
      
      st.subheader('General Stock Info') 
      st.markdown('** Market **: ' + info['market'])
      st.markdown('** Exchange **: ' + info['exchange'])
      st.markdown('** Quote Type **: ' + info['quoteType'])
      
      start = datetime.today()-timedelta(2 * 365)
      end = datetime.today()
      df = yf.download(ticker,start,end)
      df = df.reset_index()
      
      fig = go.Figure(data=go.Scatter(x=df['Date'], y=df['Adj Close']))
      fig.update_layout(
          title={
              'text': "Stock Prices Over Past Two Years",
              'y':0.9, 
              'x':0.5,
              'xanchor': 'center', 
              'yanchor': 'top'
          }
      )
      st.plotly_chart(fig, use_container_width=True)
      marketInfo = {
              "Volume": info['volume'],
              "Average Volume": info['averageVolume'],
              "Market Cap": info["marketCap"],
              "Float Shares": info['floatShares'],
              "Regular Market Price (USD)": info['regularMarketPrice'],
              'Bid Size': info['bidSize'],
              'Ask Size': info['askSize'],
              "Share Short": info['sharesShort'],
              'Short Ratio': info['shortRatio'],
              'Share Outstanding': info['sharesOutstanding']
          }
      marketDF = pd.DataFrame(data=marketInfo, index=[0])
      st.table(marketDF)
      st.write(' *'*25)
      st.subheader('- To Work In A Different Analysis Category:')
      st.write('* Go To Step #1')
      st.subheader('- To Use Other Models Within This Same Analysis Category:')
      st.write('* Go To Step #2')      

else:
  def calcMovingAverage(data, size):
      df = data.copy()
      df['sma'] = df['Adj Close'].rolling(size).mean()
      df['ema'] = df['Adj Close'].ewm(span=size, min_periods=size).mean()
      df.dropna(inplace=True)
      return df
  
  def calc_macd(data):
      df = data.copy()
      df['ema12'] = df['Adj Close'].ewm(span=12, min_periods=12).mean()
      df['ema26'] = df['Adj Close'].ewm(span=26, min_periods=26).mean()
      df['macd'] = df['ema12'] - df['ema26']
      df['signal'] = df['macd'].ewm(span=9, min_periods=9).mean()
      df.dropna(inplace=True)
      return df

  def calcBollinger(data, size):
      df = data.copy()
      df["sma"] = df['Adj Close'].rolling(size).mean()
      df["bolu"] = df["sma"] + 2*df['Adj Close'].rolling(size).std(ddof=0) 
      df["bold"] = df["sma"] - 2*df['Adj Close'].rolling(size).std(ddof=0) 
      df["width"] = df["bolu"] - df["bold"]
      df.dropna(inplace=True)
      return df


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

 
if(systemStage == 'Technical-Analysis'):
  st.title('Technical Analysis Home Page')
  st.write(' *'*25)
  st.header('General Analysis Notes')
  st.write("https://www.investopedia.com/terms/t/technicalanalysis.asp")
  st.write(' *'*25)
  st.subheader('Definition:')
  st.write('* Technical analysis is a trading discipline employed to evaluate investments and identify trading opportunities by analyzing\ statistical \
    trends gathered from trading activity, such as price movement and volume. ... Technical analysis can be used on any security with historical trading data.')
  st.write('* In finance, technical analysis is an analysis methodology for forecasting \
    the direction of prices through the study of past market data, primarily price and volume.')
  st.write("* Technical analysis is concerned with price action, which gives clues as to the stock's supply and demand dynamics\
     – which is what ultimately determines the stock price.")
  st.write(' *'*25)
  st.subheader('Examples of technical analysis tools include:')
  st.write('* All of the tools have the same purpose: to make understanding chart movements and identifying trends easier for technical traders.')
  st.write('* moving averages')
  st.write('* support and resistance levels')
  st.write('* Bollinger bands')
  st.write('* and more')
  st.write("https://www.investopedia.com/top-7-technical-analysis-tools-4773275")
  st.write(' *'*25)
  st.subheader('KEY TAKEAWAYS')
  st.write("* Technical analysis, or using charts to identify trading signals and price patterns, may seem overwhelming or esoteric at first.")
  st.write("* Beginners should first understand why technical analysis works as a window into market psychology to identify opportunities to profit.")
  st.write("* Focus on a particular trading approach and develop a disciplined strategy that you can follow without letting emotions or second-guessing get in the way.")
  st.write("* Find a broker that can help you execute your plan affordably while also providing a trading platform with the right suite of tools you'll need.")
  st.write(' *'*25)
  st.header('Model Results Below:')

  st.sidebar.subheader('> Step #2')
  ticker = st.sidebar.text_input('Enter Stock Ticker IN ALL CAPS')
  if ticker:
    st.sidebar.subheader('Ticker Input = Good')
    st.sidebar.write(' *'*25)
    import requests
    def get_symbol(symbol):
        url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
        result = requests.get(url).json()
        for x in result['ResultSet']['Result']:
            if x['symbol'] == symbol:
                return x['name']
    technical_company = get_symbol(ticker)

    st.sidebar.subheader('> Step #3')
    st.sidebar.markdown(f"Hit 'Run' For Technical Analysis On:\n\n {technical_company} ({ticker})")
    run_button = st.sidebar.button("RUN")
    if run_button:
      st.subheader('Moving Average')
      coMA1, coMA2 = st.beta_columns(2)
      with coMA1:
          numYearMA = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=0)    
      with coMA2:
          windowSizeMA = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=1)  
          
      start = datetime.today()-timedelta(numYearMA * 365)
      end = datetime.today()
      
      dataMA = yf.download(ticker,start,end)
      df_ma = calcMovingAverage(dataMA, windowSizeMA)
      df_ma = df_ma.reset_index()
      
      figMA = go.Figure()    
      figMA.add_trace(go.Scatter(x=df_ma['Date'], y=df_ma['Adj Close'], name="Prices Over Last "+str(numYearMA)+" Year(s)"))    
      figMA.add_trace(go.Scatter(x=df_ma['Date'], y=df_ma['sma'], name="SMA"+str(windowSizeMA)+" Over Last "+str(numYearMA)+" Year(s)"))
      figMA.add_trace(go.Scatter(x=df_ma['Date'], y=df_ma['ema'], name="EMA"+str(windowSizeMA)+" Over Last "+str(numYearMA)+" Year(s)"))
      figMA.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
      figMA.update_layout(legend_title_text='Trend')
      figMA.update_yaxes(tickprefix="$")
      st.plotly_chart(figMA, use_container_width=True)  
      
      st.subheader('Moving Average Convergence Divergence (MACD)')
      numYearMACD = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=2) 
      startMACD = datetime.today()-timedelta(numYearMACD * 365)
      endMACD = datetime.today()
      dataMACD = yf.download(ticker,startMACD,endMACD)
      df_macd = calc_macd(dataMACD)
      df_macd = df_macd.reset_index()
      
      figMACD = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)
      figMACD.add_trace(go.Scatter(x = df_macd['Date'], y = df_macd['Adj Close'], name = "Prices Over Last " + str(numYearMACD) + " Year(s)"), row=1, col=1)
      figMACD.add_trace(go.Scatter(x = df_macd['Date'], y = df_macd['ema12'], name = "EMA 12 Over Last " + str(numYearMACD) + " Year(s)"), row=1, col=1)
      figMACD.add_trace(go.Scatter(x = df_macd['Date'], y = df_macd['ema26'], name = "EMA 26 Over Last " + str(numYearMACD) + " Year(s)"), row=1, col=1)
      figMACD.add_trace(go.Scatter(x = df_macd['Date'], y = df_macd['macd'], name = "MACD Line"), row=2, col=1)
      figMACD.add_trace(go.Scatter(x = df_macd['Date'], y = df_macd['signal'], name = "Signal Line"), row=2, col=1)
      figMACD.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=0))
      figMACD.update_yaxes(tickprefix="$")
      st.plotly_chart(figMACD, use_container_width=True)
      
      st.subheader('Bollinger Band')
      coBoll1, coBoll2 = st.beta_columns(2)
      with coBoll1:
          numYearBoll = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=6) 
      with coBoll2:
          windowSizeBoll = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=7)
          
      startBoll= datetime.today()-timedelta(numYearBoll * 365)
      endBoll = datetime.today()
      dataBoll = yf.download(ticker,startBoll,endBoll)
      df_boll = calcBollinger(dataBoll, windowSizeBoll)
      df_boll = df_boll.reset_index()
      
      figBoll = go.Figure()
      figBoll.add_trace(go.Scatter(x = df_boll['Date'], y = df_boll['bolu'], name = "Upper Band"))
      figBoll.add_trace(go.Scatter(x=df_boll['Date'], y=df_boll['sma'], name="SMA"+str(windowSizeBoll)+" Over Last "+str(numYearBoll)+" Year(s)"))
      figBoll.add_trace(go.Scatter(x = df_boll['Date'], y = df_boll['bold'], name = "Lower Band"))
      figBoll.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=0))
      figBoll.update_yaxes(tickprefix="$")
      st.plotly_chart(figBoll, use_container_width=True)
      st.write(' *'*25)
      st.subheader('- To Work In A Different Analysis Category:')
      st.write('* Go To Step #1')
      st.subheader('- To Use Other Models Within This Same Analysis Category:')
      st.write('* Go To Step #2')


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if(systemStage == 'Forecasting'):
  st.title('Forecasting Price Points - Home Page')
  st.subheader("Use The Side Bar via the Arrow ('>') on the upper left corner of the screen")

  models = [
    '-Select-A-Model-', 'Prophet Model', 'Stocker Analysis', 'A.R.I.M.A', 'S.A.R.I.M.A', 
    'Monte Carlo Simulation', 'Univariate Analysis'
    ]
  
  st.sidebar.header('[Step # 2]')
  st.sidebar.subheader('Select Model To Run')
  model = st.sidebar.selectbox('Model List:', models)
  st.sidebar.write(' *'*25)  

  st.sidebar.header('[Step # 3]')
  st.sidebar.subheader(f'Pick A Stock For {model} & Review')
  stock_ticker = st.sidebar.text_input('Enter A Stock Ticker By Typing All Caps Into The Below Box')
  st.sidebar.write(' * example: TSLA ')
  st.sidebar.write(' *'*25)


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model == 'Prophet Model'):
    st.title('Prophet Model Forecasting')
    st.markdown(' * Prophet is a procedure for forecasting time series data based on an additive model where \
    non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects.')
    st.markdown(' * It works best with time series that have strong seasonal effects and several seasons of historical data. \
    Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.')
    st.markdown(f" * web address = {'https://facebook.github.io/prophet/docs/quick_start.html#python-api'}")
    st.header(' *'*34)

    st.sidebar.subheader('Select The Forcast Horizon')
    forcast_horizon = st.sidebar.radio('Forcast Range Options', [180,360])
    st.sidebar.write(' *'*25)

    st.sidebar.header('WHICH MOD')
    which_mod = st.sidebar.radio('which_mod', ['normal','full_run'])

    if which_mod == 'normal':
      st.sidebar.write(' *'*25)
      st.sidebar.header('[Step # 3] - Run The Model')
      staged_button = st.sidebar.button(f"Configure-{model.capitalize()}-Model")
      st.sidebar.write(' *'*25)
      if staged_button:
        f1.Web_prophet_kyle(stock_ticker).make_forecast1(stock_ticker,forcast_horizon, '2y')
        st.title('Model Render Complete')

    if which_mod == 'full_run':
      st.sidebar.write(' *'*25)      
      st.sidebar.header('[Step # 3] - Run The Model')
      staged_button = st.sidebar.button(f"Configure-{model.capitalize()}-Model")
      st.sidebar.write(' *'*25)
      if staged_button:
        f1.Web_prophet_kyle(stock_ticker).make_forecast1(stock_ticker,forcast_horizon, '2y')
        st.header('Model-A Render Complete')     
        f1.Web_prophet_kyle(stock_ticker).make_forecast2(stock_ticker,forcast_horizon)        
        st.title('Model-B Render Complete')


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='Stocker Analysis'):
    st.title('STOCKER MODELING')

    if stock_ticker:
      run_strategy_stocker = st.sidebar.button("Run Stocker")
      if run_strategy_stocker:
        f1.web_stocker_run(stock_ticker)
        st.title('Model Render Complete')


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='A.R.I.M.A'):
    st.title('(A.R.I.M.A)')
    st.header('Auto Regression Integrated Moving Average')

    if stock_ticker:
      run_strategy_arima = st.button("Run ARIMA")
      if run_strategy_arima:
        f1.Web_Arima(stock_ticker).full_build()
        f1.The_Arima_Model(stock_ticker).arima_model()


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='S.A.R.I.M.A'):
    st.title('(S.A.R.I.M.A)')
    st.header('Seasonal AutoRegressive Integrated Moving Average')
    st.subheader('SARIMA MODELING & FORECAST')
    st.write(' *'*34)
    st.markdown("- A Seasonal-AutoRegressive-Integrated-MovingAverage (SARIMA) model is 1 step more than \
      an ARIMA model based on the concept of seasonal trends")
    st.markdown("- In many time series data, frequent seasonal effects come into play.")
    st.markdown("- Take for example the average temperature measured in a location with four seasons.")
    st.write(' *'*34)

    st.header('Null Hypothesis (Ho)')
    st.subheader('[FAIL-TO-REJECT] ')
    st.write(' * suggests the TimeSeries contains a unit-root ')
    st.write(' ---> NON-STATIONARY')
    st.header('Alternate Hypothesis (Ha)')
    st.subheader('[REJECT] ')
    st.write(' * suggests the TimeSeries does NOT contain a unit-root')
    st.write(' ---> STATIONARY')
    st.write(' *'*34)

    st.header('Rules To Accept or Reject Null Hypothesis (Ho) - Using the p-value')
    st.subheader(f"[REJECT] ")
    st.write("- the null IF the p-value falls below a threshold value such as 5% or even 1%")
    st.subheader(f"[FAIL-TO-REJECT] ")
    st.write("- the null if the p-value is above the threshold value")
    st.write(f" * Once [FAIL-TO-REJECT] standard is met, can consider the time series as NON-STATIONARY")
    st.write(' *'*34)

    st.header('Examples:')
    st.subheader("( p-value > 0.05 ) == [FAIL-TO-REJECT]")
    st.write(" * [Conclude data has unit-root]")
    st.write(" * [NON-STATIONARY]")
    st.subheader("( p-value =< 0.05 ) == [REJECT NULL HYPOTHESIS]")
    st.write(" * [Conclude data does NOT contain unit-root]")
    st.write(" * [STATIONARY]")
    st.write(' *'*34)

    if stock_ticker:
      run_strategy_sarima = st.sidebar.button("Run SARIMA")
      if run_strategy_sarima:
        f1.sarima(stock_ticker).predict()
        st.title('Model Render Complete')


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model == 'Monte Carlo Simulation'):
    st.title('Monte Carlo Simulations')
    st.markdown(' * A Monte Carlo simulation is a useful tool for predicting future results calculating \
    a formula multiple times with different random inputs.')
    st.markdown(' * This is a process you can execute in Excel but it is not simple to do without some VBA \
      or potentially expensive third party plugins.')

    if stock_ticker:
      run_strategy_monteCarlo = st.sidebar.button("Run Monte Carlo")
      if run_strategy_monteCarlo:
        f1.monteCarlo(stock_ticker)
        st.title('Model Render Complete')      


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='Univariate Analysis'):
    st.title('UNIVARIATE TIME-SERIES MODELING & FORCASTS')
    st.write("The term 'univariate' implies that forecasting is based on a sample of time series observations of the \
      exchange rate without taking into account the effect of the other variables such as prices and interest rates.\
        If this is the case, then there is no need to take an explicit account of these variables."
    )
    if stock_ticker:
      f1.univariate(stock_ticker).runs()
    else:
      ticker_univariate = st.text_input('Enter Ticker For Univariate Model:')
      if ticker_univariate:
        run_strategy_univariate = st.sidebar.button("Run Univariate")
        if run_strategy_univariate:
          f1.univariate(ticker_univariate).runs()
          st.title('Model Render Complete')


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if(systemStage=='Strategy'): 
  st.title('Strategy Home Page')
  st.write(' *'*25)
  st.header('> Site Navigation:')
  st.write("\n* Use The Side Bar via the Arrow ('>') on the upper left corner of the screen")
  st.subheader('~ To Work In A Different Analysis Category:')
  st.write('* Go To Step #1')
  st.subheader('~ To Use Other Models Within This Same Analysis Category:')
  st.write('* Go To Step #2')
  st.write(' *'*25)
  
  st.header('> General Analysis Components')

  st.subheader('Moving Averages')
  st.write('* Double Moving Averages')
  st.write('* Exponential Moving Average (EMA)')
  st.write('* Simple Moving Average (SMA)')
  st.write('* Bollinger Bands')
  st.write('* MOM')
  st.write('* MACD')
  st.write('* RSI')
  st.write('* APO')

  st.subheader('Regression')
  st.write('* Linear Regression')
  st.write('* Quadratic Regression 2 & 3')
  st.write('* KNN')
  st.write('* Lasso')      
  st.write('* Ridge')
  st.write('* Logistic Regression')

  st.subheader('Speciality Trading')
  st.write('* naive momentum')
  st.write('* Pairs Correlation Trading')
  st.write('* Support & Resistance')
  st.write('* Turtle Trading')
  st.write('* Mean Reversion & Trend Following')          
  st.write('* Volatility Mean Reversion & Trend Following')
  st.write('* OverBought & OverSold')

  st.subheader('Strategy Backtesting')
  st.write('* xgboost sim/backtesting')
  st.write('* backtrader backtesting')
  
  models = [
    '-Select-Model-','Moving Averages - SMA & EMA','Moving Averages - B','Support & Resistance Lines',
    'overBought_overSold','Backtrader - SMA Strategy','BackTesting - 1'
  ]

  st.sidebar.subheader('> Step #2')
  model = st.sidebar.selectbox('Choose A Model', models)
  st.sidebar.write(' *'*25)

  st.sidebar.subheader('> Step #3')
  stock_ticker = st.sidebar.text_input('Type In Stock Ticker To Model (ALL CAPS): ')
  st.sidebar.write(' * example: TSLA ')
  st.sidebar.write(' *'*25)

  import requests
  def get_symbol(symbol):
      url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
      result = requests.get(url).json()
      for x in result['ResultSet']['Result']:
          if x['symbol'] == symbol:
              return x['name']
  strategy_company = get_symbol(stock_ticker)


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='Moving Averages - SMA & EMA'):
    st.write(' *'*25)
    st.title('Moving Averages - SMA & EMA')
    fin = False

    if stock_ticker:
      st.sidebar.subheader('> Step #4')
      st.sidebar.write('Select The Short & Long Moving Average Intervals:')
      short_SMA_EMA = int(st.sidebar.number_input('Enter A SHORT Moving Average:'))
      if short_SMA_EMA:
        long_SMA_EMA = int(st.sidebar.number_input('Enter A LONG Moving Average:'))
        st.sidebar.write(' *'*25)

        if long_SMA_EMA:
          st.sidebar.subheader('> Step #5')
          st.sidebar.write('Click Button Below To Run Model')
          run_strategy_movAvg_SMA_EMA = st.sidebar.button("Run Moving Average - SMA & EMA")
          if run_strategy_movAvg_SMA_EMA:
            f2.MovingAverageCrossStrategy(
              stock_symbol = stock_ticker, 
              start_date = '2020-01-01', 
              end_date = '2021-04-16', 
              short_window = short_SMA_EMA, 
              long_window = long_SMA_EMA, 
              moving_avg = 'SMA', 
              display_table = True
            )
            st.subheader(f'The Above Table Is A Record Of Buy & Sell Signals For: \n {strategy_company} ({stock_ticker})')
            st.write(f'* Using the short-{short_SMA_EMA} & long-{long_SMA_EMA} Moving Average Intervals.')
            f2.MovingAverageCrossStrategy(
              stock_symbol = stock_ticker, 
              start_date = '2020-01-01', 
              end_date = '2021-04-16', 
              short_window = short_SMA_EMA, 
              long_window = long_SMA_EMA, 
              moving_avg = 'EMA', 
              display_table = True
            )
            st.subheader(f'The Above Table Is A Record Of Buy & Sell Signals For: \n {strategy_company} ({stock_ticker})')
            st.write(f'* Using the short-{short_SMA_EMA} & long-{long_SMA_EMA} Moving Average Intervals.')
            fin = True

    if fin:
      st.write(' *'*25)
      st.title('Model Render Complete')      


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='Moving Averages - B'):
    st.title('Moving Average 0')
    fin = False

    if stock_ticker:
      run_strategy_movAvg_B = st.sidebar.button("Run Moving Average-B (Double)")
      if run_strategy_movAvg_B:
        f2.ST_Trading_signals(stock_ticker)
        fin = True

    if fin:
      st.write(' *'*25)
      st.title('Model Render Complete')        


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='Support & Resistance Lines'):
    st.title('Support & Resistance Lines')
    fin = False

    if stock_ticker:
      run_strategy_supportResistance = st.sidebar.button("Run Support & Resistance Lines")
      if run_strategy_supportResistance:
        f2.The_Support_Resistance(stock_ticker, '6mo').level()
        fin = True

    if fin:
      st.write(' *'*25)
      st.title('Model Render Complete')


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='overBought_overSold'):
    st.title('Over Bought & Over Sold Analysis')
    fin = False

    if stock_ticker:
      run_strategy_overBought_overSold = st.sidebar.button("Run Over-Bought & OverSold")
      if run_strategy_overBought_overSold:
        f2.The_OverBought_OverSold(stock_ticker, '1y').generate()
        fin = True

    if fin:
      st.write(' *'*25)
      st.title('Model Render Complete')      


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  if(model=='Backtrader - SMA Strategy'):
    fin = False
    st.title('Backtrader For Testing - SMA Strategy')
    st.write('details')

    if stock_ticker:
      run_strategy_backtraderSMA = st.sidebar.button("Run Backtrader SMA Strategy")
      if run_strategy_backtraderSMA:
        f2.backtrader_sma_strategy_run(stock_ticker)
        fin = True

    if fin:
      st.write(' *'*25)
      st.title('Model Render Complete')        
      

# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='BackTesting - 1'):
    fin = False
    st.title('BackTesting - 1')
    st.write('details')

    if stock_ticker:
      run_strategy_backtesting1 = st.sidebar.button("Run Backtest 1")
      if run_strategy_backtesting1:
        f2.Web_One(stock_ticker)
        fin = True

    if fin:
      st.write(' *'*25)
      st.title('Model Render Complete')        


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if(systemStage=='Portfolio'):
  st.title('Portfolio Allocation & Optimization')
  st.write(' *'*25)

  st.header('> Site Navigation:')
  st.write("\n* Use The Side Bar via the Arrow ('>') on the upper left corner of the screen")
  st.subheader('~ To Work In A Different Analysis Category:')
  st.write('* Go To Step #1')
  st.subheader('~ To Use Other Models Within This Same Analysis Category:')
  st.write('* Go To Step #2')
  st.write(' *'*25)
  
  st.title('> General Analysis Definitions')

  st.header('1) Principal Component Analysis (PCA)')
  st.write('* Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of \
    large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.')
  st.write('* https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c')
  st.write(' *'*25)

  st.header('2) Markowitz Efficient Frontier')
  st.write("* In modern portfolio theory, the efficient frontier is an investment portfolio which occupies the 'efficient'\
     part of the risk–return spectrum.\ Formally, it is the set of portfolios which satisfy the condition that no other portfolio \
       exists with a higher expected return but with the same standard deviation of return.")
  st.write("* The efficient frontier is the set of optimal portfolios that offer the highest expected return for a \
      defined level of risk or the lowest risk for a given level of expected return. Portfolios that lie below the \
        efficient frontier are sub-optimal because they do not provide enough return for the level of risk.")
  st.write('* https://www.investopedia.com/terms/e/efficientfrontier.asp')
  st.write(' *'*25)

  st.header('3) Modern Portfolio Theory Portfolio Optimization')
  st.write('* Portfolio optimization is the process of selecting the best portfolio (asset distribution), out of the set of all portfolios\
     being considered, according to some objective. The objective typically maximizes factors such as expected return, and minimizes costs like financial risk.')
  st.write('* Modern portfolio theory (MPT) is a theory on how risk-averse investors can construct portfolios\
      to maximize expected return based on a given level of market risk. Harry Markowitz pioneered this theory in his \
        paper "Portfolio Selection," which was published in the Journal of Finance in 1952.')
  st.subheader('Key Assumptions of Modern Portfolio Theory')
  st.write('* At the heart of MPT is the idea that risk and return are directly linked. \
    This means that an investor must take on a higher level of risk to achieve greater expected returns.')
  st.write('* https://www.investopedia.com/terms/m/modernportfoliotheory.asp')
  st.write(' *'*25)


  models = ['-Select-Model-', 'Principal Component Analysis', 'Efficient Frontier', 'Portfolio Optimizer']
  st.sidebar.subheader('> Step # 2')
  model = st.sidebar.selectbox('Choose A Model', models)
  st.sidebar.write(' *'*25)


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='Principal Component Analysis'):
    st.title('(1) Principal Component Analysis (PCA)')
    st.header('Model Results Below:')
    fin = False

    st.sidebar.subheader('> Step #3')
    tickers = st.sidebar.selectbox('Choose Stock List', index_ticker_lists_B)
    if tickers:
      for idx, num in enumerate(index_ticker_lists_B):
        if num == tickers:
          new_tickers = index_ticker_lists_A[idx]
          lst_name = num
      
      st.sidebar.subheader("This Ticker List Contains The Following Stock Tickers:")
      st.sidebar.markdown(new_tickers)
      st.sidebar.write(' *'*25)
      st.sidebar.subheader('> Step #4')
      st.sidebar.markdown("Hit The 'Run PCA' Button To Run Model")
      run_strategy_pca = st.sidebar.button("Run PCA")
      if run_strategy_pca:
        f3.The_PCA_Analysis(new_tickers, lst_name)
        fin = True

    if fin:
      st.write(' *'*25)
      st.title('Model Render Complete')


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='Efficient Frontier'):
    st.title('(2) Efficient Frontier')
    st.header('Model Results Below:')
    fin = False

    st.sidebar.subheader('> Step #3A')
    EF_portfolio = st.sidebar.selectbox('Select To Use Pre-Built 10 Security List Or Design Own Portfolio', ['Pre-Built','My-Own-Portfolio'])
    st.sidebar.markdown('This is where you can select Pre-Built and enter in the tickers from the PCA Analysis')
    
    st.sidebar.subheader('> Step #3B')
    if EF_portfolio == 'Pre-Built':
      RISKY_ASSETS = st.sidebar.text_input('Enter Ticker List Here:')
      RISKY_ASSETS = RISKY_ASSETS.split()
      st.sidebar.text(RISKY_ASSETS)
      if RISKY_ASSETS:
        RISKY_ASSETS.sort()
        marks0 = ['o', '^', 's', 'p', 'h', '8','*', 'd', '>', 'v', '<', '1', '2', '3', '4']
        mark = marks0[:len(RISKY_ASSETS)+1]
        st.sidebar.write(' *'*10)
        st.sidebar.subheader('> Step #4')
        run_strategy_EF = st.sidebar.button("Run Efficient Frontier")
        if run_strategy_EF:
          f3.The_Efficient_Frontier(RISKY_ASSETS, mark).final_plot()
          fin = True
 
    if EF_portfolio == 'My-Own-Portfolio':
      manys = [2,4,6,8,10,12,14]
      num_stocks = int(st.sidebar.selectbox('Select Number Of Securities For Portfolio:',manys))    
      if num_stocks:
        RISKY_ASSETS = []
        for n in range(1,num_stocks+1):
          tic = st.text_input(f'Ticker {n}: ')
          RISKY_ASSETS.append(tic)
        RISKY_ASSETS.sort()
        marks0 = ['o', '^', 's', 'p', 'h', '8','*', 'd', '>', 'v', '<', '1', '2', '3', '4']
        mark = marks0[:len(RISKY_ASSETS)+1]
        run_strategy_EF = st.button("Run Efficient Frontier")
        if run_strategy_EF:
          f3.The_Efficient_Frontier(RISKY_ASSETS, mark).final_plot()
          fin = True

    if fin:
      st.write(' *'*25)
      st.title('Model Render Complete')


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='Portfolio Optimizer'):
    st.title('Portfolio Optimizer')
    st.header('Model Results Below:')
    fin = False

    st.sidebar.subheader('> Step #3')
    Em = str(st.sidebar.selectbox('Pick Ticker Lists:',['Pick-Em','Pick From Ticker Lists']))

    if Em:

      if Em == 'Pick-Em':
        stock_tickers = st.sidebar.text_input('Enter Ticker List Here: (ex. DIS ECL PLNT NYT)')
        stock_tickers = stock_tickers.split()
        if type(stock_tickers)==list:
          st.sidebar.subheader('ticker list entered in good order')
          st.sidebar.markdown(stock_tickers)
          st.sidebar.write(' *'*25)
          st.sidebar.subheader('> Step #4 - Run Optimization')
          buttonA = st.sidebar.button('Run Optimizer A')
          if buttonA:
            f3.The_Portfolio_Optimizer(stock_tickers, 'Pick_EM_Portfolio').optimize()
            fin = True

      if Em == 'Pick From Ticker Lists':
        stockS = st.sidebar.selectbox('Choose Ticker List: ', index_ticker_lists_B)
        for idx, num in enumerate(index_ticker_lists_B):
          if num == stockS:
            st.sidebar.subheader('> Step #4 - Run Optimization')
            buttonB = st.sidebar.button('Run Optimizer B')
            if buttonB:
              f3.The_Portfolio_Optimizer(index_ticker_lists_A[idx], num).optimize()
              fin = True

    if fin:
      st.write(' *'*25)
      st.title('Model Render Complete')
       

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *