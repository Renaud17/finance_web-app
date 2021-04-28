import requests
import pandas as pd 
from yahoo_fin import stock_info as si 
import yfinance as yf
from pandas_datareader import DataReader
import numpy as np
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
from pathlib import Path
import pickle


class Recommendations1(object):
    def __init__(self, ticker_list, ticker_list_name):
        self.tickers = ticker_list
        self.name = ticker_list_name


    def run_rec1(self):
        recommendations = []
        for ticker in self.tickers:
            lhs_url = 'https://query2.finance.yahoo.com/v10/finance/quoteSummary/'
            rhs_url = '?formatted=true&crumb=swg7qs5y9UP&lang=en-US&region=US&' \
                    'modules=upgradeDowngradeHistory,recommendationTrend,' \
                    'financialData,earningsHistory,earningsTrend,industryTrend&' \
                    'corsDomain=finance.yahoo.com'
            url =  lhs_url + ticker + rhs_url
            r = requests.get(url)
            if not r.ok:
                recommendation = 6
            try:
                result = r.json()['quoteSummary']['result'][0]
                recommendation =result['financialData']['recommendationMean']['fmt']
            except:
                recommendation = 6
            recommendations.append(recommendation)
            
        dataframe = pd.DataFrame(list(zip(self.tickers, recommendations)), columns =['Company', 'Recommendations']) 
        dataframe = dataframe.set_index('Company')
        dataframe['Recommendations']=dataframe['Recommendations'].astype(float)
        dataframe = dataframe.sort_values('Recommendations')
        st.header(f'Analyst Recommendations - {self.name}')
        st.dataframe(dataframe)

        dataframe = dataframe[dataframe['Recommendations'] < 2.5]
        recomendation_ticker_list = list(dataframe.index)
        st.subheader("The Below List contains Only The Stocks With A 'Buy' or 'Strong-Buy' Rating")
        st.write(f"* Total Stocks In Buy/Strong-Buy Territory = {len(recomendation_ticker_list)} / {len(self.tickers)}")
        st.text(recomendation_ticker_list)
        st.write(' *'*25)
        return recomendation_ticker_list


if __name__ =='__main__':

    dow_ticker_lst = pd.read_pickle('tickers/dow_ticker_lst.pkl')
    sp100_ticker_lst = pd.read_pickle('tickers/sp100_ticker_lst.pkl')
    sp500_ticker_lst = pd.read_pickle('tickers/sp500_ticker_lst.pkl')
    chuck_merged_ticker_lst = pd.read_pickle('tickers/chuck_merged_ticker_lst.pkl')
    jayci_ticker_lst = pd.read_pickle('tickers/jayci_ticker_lst.pkl')
    watch_merged_ticker_lst = pd.read_pickle('tickers/watch_merged_ticker_lst.pkl')

    all_ticker_lists = [
        dow_ticker_lst, sp100_ticker_lst, sp500_ticker_lst, 
        chuck_merged_ticker_lst, jayci_ticker_lst, watch_merged_ticker_lst
    ]
    all_ticker_list_names = [
        'DOW Ticker List', 'SP100 Ticker List', 'SP500 Ticker List', 
        'Chuck Merged Ticker List', 'Jayci Ticker List', 'Watching Ticker List'
    ]    

    run = False

    if run:
        for a in range(len(all_ticker_lists)):
            Recommendations1(all_ticker_lists[a], all_ticker_list_names[a]).run_rec1()