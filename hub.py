#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
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


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


from pages import forecast as f1   # web_monteCarlo, web_prophet, web_sarima
from pages import strategy as f2   # BT_SmaStrategy, 
from pages import portfolio as f3


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


saveTickers = Path('tickers/')

indices_dow = ['^DJI','^DJT','^DJU','^DJA']                  # Industrial, Transportation, Utility, Composite
indices_sp = ['^GSPC','^OEX','^MID','^SP1000','^SP1500']     # SP500, SP100, SP400, SP400-MID, SP1000, SP1500
indices_nasdaq = ['^IXCI','^NDX','^IXCO','^IXHC','^IXF']     # NASDAQ, COMP, NASDAQ-100, Computer, HC, Fin100
indices_nyse = ['^NYA','^PSE','^XAX','^XMI']                 # NYSEComp, ArcaTech100, AMEXComp, ArcaMajorMkt
indices_russell = ['^RUA','^RUT','RUI']                      # Russell-3k, Russell-2k, Russell-1k
indices_foreign = ['^FTSE','^GDAXI']                         # FTSE-100, DAX-Performance
indices_cboe = ['^VIX','^TNX','TYX']                         # CBOE-Vol, 10-YrT, 30-YrT
track_sp500 = ['SPY','SPLG','IVV','VOO']                     # SPDR-etf, SPDRPort-etf, iSharesCore, Van-etf
track_russell = ['IWM','VTWO','URTY']                        # iShareRuss2k, VanRuss2k, ProSharesRuss2k
track_total_mkt = ['VTSAX', 'FSKAX','SWTSX','IWV']           # VanTtlMkt, FidTtlMkt, SchwabTtlMkt, iShRus3k
track_vanguard = ['VFINX','VFIAX','VIMAX','VTSAX','VSMAX']   # Inv, Adm500, MID-Adm, TtlMkt-Adm, SmCapAdm  

dow = pd.read_pickle(saveTickers / f'dow_ticker_lst.pkl')
sp100 = pd.read_pickle(saveTickers / f'sp100_ticker_lst.pkl')
sp400 = pd.read_pickle(saveTickers / f'sp400_ticker_lst.pkl')
sp500 = pd.read_pickle(saveTickers / f'sp500_ticker_lst.pkl')
sp600 = pd.read_pickle(saveTickers / f'sp600_ticker_lst.pkl')
sp1000 = pd.read_pickle(saveTickers / f'sp1000_ticker_lst.pkl')
nasdaq_lst = pd.read_pickle(saveTickers / f'nasdaq_ticker_lst.pkl')
other_lst = pd.read_pickle(saveTickers / f'other_ticker_lst.pkl')

indices_main = ['^OEX','^MID','^GSPC','^DJI','^NYA','^RUT','^W5000']
index_names = ['SP100','SP400','SP500','DOW','NYSE','Russ2k','Wilshire5k']
combined_index_main_names = [list(x) for x in zip(indices_main, index_names)]

my_positions = pd.read_pickle(saveTickers / f'chuck_merged_ticker_lst.pkl')
watch_lst = pd.read_pickle(saveTickers / f'watch_merged_ticker_lst.pkl')
my_tickers = my_positions + watch_lst


index_ticker_lists_A = [
  dow, sp100, sp400, sp500, indices_main, my_tickers, my_positions, watch_lst,
  indices_dow, indices_sp, indices_nasdaq, indices_nyse, indices_russell, indices_foreign, indices_cboe,
  track_sp500, track_russell, track_total_mkt, track_vanguard
]
index_ticker_lists_B = [
  'dow', 'sp100', 'sp400', 'sp500', 'indices_main', 'my_tickers', 'my_positions', 'watch_lst',
  'indices_dow', 'indices_sp', 'indices_nasdaq', 'indices_nyse', 'indices_russell', 'indices_foreign', 'indices_cboe',
  'track_sp500', 'track_russell', 'track_total_mkt', 'track_vanguard'
]


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


st.progress(0)
st.balloons()

systemStage = st.selectbox(
  '(Action # 1) - Select The System Stage', 
  [
    '-Select-Stage-','Forecasting','Strategy', 'Portfolio'
    ]
)

if(systemStage=='-Select-Stage-'):
  st.title('Fun Forecasting For Friends')
  st.subheader('* Select A Stage Then Use the  Side Bar to:')


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if(systemStage == 'Forecasting'):
  st.subheader("Use The Side Bar via the Arrow ('>') on the upper left corner of the screen")

  models = [
    '-Select-A-Model-', 'Prophet Model', 'Stocker Analysis', 'A.R.I.M.A', 'S.A.R.I.M.A', 
    'Monte Carlo Simulation', 'Univariate Analysis'
    ]
    # 'multivariate','linearRegression','quadraticRegression-2','quadraticRegression-3','knn-regression']
  
  st.sidebar.header('[Step # 2]')
  st.sidebar.subheader('Select Model To Run')
  model = st.sidebar.selectbox('Model List:', models)
  st.sidebar.write(' *'*25)  

  st.sidebar.header('[Setp # 3]')
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
    forcast_horizon = st.sidebar.radio('Forcast Range Options', [180,360,720])
    st.sidebar.write(' *'*25)

    st.sidebar.header('WHICH MOD')
    which_mod = st.sidebar.radio('which_mod', ['normal','full_run'])

    if which_mod == 'normal':
      st.sidebar.write(' *'*25)
      st.sidebar.header('[Step # 3] - Run The Model')
      staged_button = st.sidebar.button(f"Configure-{model.capitalize()}-Model")
      st.sidebar.write(' *'*25)
      if staged_button:
        historical_data='2Y'
        f1.Web_prophet_kyle(stock_ticker).make_forecast1(stock_ticker,forcast_horizon, historical_data)
        st.title('Model Render Complete')

    if which_mod == 'full_run':
      st.sidebar.write(' *'*25)      
      st.sidebar.header('[Step # 3] - Run The Model')
      staged_button = st.sidebar.button(f"Configure-{model.capitalize()}-Model")
      st.sidebar.write(' *'*25)
      if staged_button:
        historical_data='2Y'
        f1.Web_prophet_kyle(stock_ticker).make_forecast1(stock_ticker,forcast_horizon, historical_data)
        st.header('Model-A Render Complete')     
        f1.Web_prophet_kyle(stock_ticker).make_forecast2(stock_ticker,forcast_horizon)        
        st.title('Model-B Render Complete')


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='Stocker Analysis'):
    st.title('STOCKER MODELING')

    run_strategy = st.sidebar.button("Run")

    if stock_ticker:
      f1.web_stocker_run(stock_ticker)
      st.title('Model Render Complete')


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='A.R.I.M.A'):
    st.title('(A.R.I.M.A)')
    st.header('Auto Regression Integrated Moving Average')

    run_strategy = st.sidebar.button("Run")

    if stock_ticker:
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

    run_strategy = st.sidebar.button("Run")

    if stock_ticker:
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

    run_strategy = st.sidebar.button("Run")

    if stock_ticker:
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

    run_strategy = st.sidebar.button("Run")

    if stock_ticker:
      f1.univariate(stock_ticker).runs()
    else:
      ticker_univariate = st.text_input('Enter Ticker For Univariate Model:')
      if ticker_univariate:
        f1.univariate(ticker_univariate).runs()
        st.title('Model Render Complete')


#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if(systemStage=='Strategy'):
  st.subheader("Use The Side Bar via the Arrow ('>') on the upper left corner of the screen")
  
  models = [
    '-Select-Model-','Moving Averages - SMA & EMA','Moving Averages - B','Support & Resistance Lines','overBought_overSold',
    'Backtrader - SMA Strategy','BackTesting - 1'
  ]
  model = st.sidebar.selectbox('(Action # 2) - Choose A Model', models)

  stock_ticker = st.sidebar.text_input('(Action # 3) - Type In Stock Ticker To Model: ')
  st.sidebar.write(' * example: TSLA ')
  run_strategy = st.sidebar.button("Run")


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  

  if(model=='Backtrader - SMA Strategy'):
    st.title('Backtrader For Testing - SMA Strategy')
    st.write('details')

    if stock_ticker:
      f2.backtrader_sma_strategy_run(stock_ticker)
      

# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='BackTesting - 1'):
    st.title('BackTesting - 1')
    st.write('details')

    if stock_ticker:
      f2.Web_One(stock_ticker)


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

  
  if(model=='Moving Averages - SMA & EMA'):
    st.title('Moving Averages - SMA & EMA')

    if stock_ticker:
      f2.MovingAverageCrossStrategy(
        stock_symbol = stock_ticker, 
        start_date = '2019-01-01', 
        end_date = '2021-04-01', 
        short_window = 20, 
        long_window = 50, 
        moving_avg = 'SMA', 
        display_table = True
      )

      f2.MovingAverageCrossStrategy(
        stock_symbol = stock_ticker, 
        start_date = '2019-01-01', 
        end_date = '2021-04-01', 
        short_window = 20, 
        long_window = 50, 
        moving_avg = 'EMA', 
        display_table = True
      )      


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='Moving Averages - B'):
    st.title('Moving Average 0')

    if stock_ticker:
      f2.ST_Trading_signals(stock_ticker)


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='Support & Resistance Lines'):
    st.title('Support & Resistance Lines')

    if stock_ticker:
      f2.The_Support_Resistance(stock_ticker, '6mo').level()


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='overBought_overSold'):
    st.title('Over Bought & Over Sold Analysis')

    if stock_ticker:
      f2.The_OverBought_OverSold(stock_ticker, '1y').generate()


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if(systemStage=='Portfolio'):
  st.subheader("Use The Side Bar via the Arrow ('>') on the upper left corner of the screen")
  
  models = [
    '-Select-Model-','Efficient Frontier', 'Principal Component Analysis', 'Portfolio Optimizer'
  ]
  model = st.sidebar.selectbox('(Action # 2) - Choose A Model', models)


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='Efficient Frontier'):
    st.title('Efficient Frontier')

    num_stocks = int(st.number_input('Enter a number'))

    if num_stocks:
      RISKY_ASSETS = []
      for n in range(1,num_stocks+1):
        tic = st.text_input(f'Ticker {n}: ')
        RISKY_ASSETS.append(tic)
      RISKY_ASSETS.sort()

      marks0 = ['o', '^', 's', 'p', 'h', '8','*', 'd', '>', 'v', '<', '1', '2', '3', '4']
      mark = marks0[:len(RISKY_ASSETS)+1]

      run_strategy = st.button("Run")
      if run_strategy:
        f3.The_Efficient_Frontier(RISKY_ASSETS, mark).final_plot()


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='Principal Component Analysis'):
    st.title('Principal Component Analysis')

    tickers = st.sidebar.selectbox('Choose Stock List', index_ticker_lists_B)

    if tickers:
      for idx, num in enumerate(index_ticker_lists_B):
        if num == tickers:
          f3.The_PCA_Analysis(index_ticker_lists_A[idx])


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


  if(model=='Portfolio Optimizer'):
    st.title('Portfolio Optimizer')


    st.sidebar.subheader('Ticker Lists:')
    pickEm = st.sidebar.checkbox('Pick-Em')
    pickLISTS = st.sidebar.checkbox('Pick From Ticker Lists')
    
    if pickEm:
      stock_tickers = []
      how_many = int(st.sidebar.text_input('How Many:'))
      for i in range(how_many):
        stock_tickers.append(st.sidebar.text_input(f'Ticker #{i}: '))
      buttonA = st.sidebar.button('Run Optimizer A')
      if buttonA:
        f3.The_Portfolio_Optimizer(stock_tickers, 'Pick_EM_Portfolio').optimize()


    elif pickLISTS:
      stockS = st.sidebar.selectbox('(Action # 2) - Choose Ticker List: ', index_ticker_lists_B)
      for idx, num in enumerate(index_ticker_lists_B):
        if num == stockS:
          buttonB = st.sidebar.button('Run Optimizer B')
          if buttonB:
            f3.The_Portfolio_Optimizer(index_ticker_lists_A[idx], num).optimize()


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *