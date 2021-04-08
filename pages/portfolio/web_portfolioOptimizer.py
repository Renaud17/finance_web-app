# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


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
plt.rcParams['figure.dpi'] = 250

import numpy as np
import pandas as pd
pd.options.display.max_rows = 999
pd.get_option("display.max_rows")
from datetime import datetime
import yfinance as yf
import scipy.optimize as sco
from pandas.io.pickle import read_pickle
import pickle
np.random.seed(777)

from pathlib import Path
from datetime import datetime

import streamlit as st
from scipy.stats import spearmanr
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


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


class The_Portfolio_Optimizer(object):
    def __init__(self, port_tics, saveName0):
        self.port_tics = port_tics
        self.saveName0 = saveName0

    def optimize(self):
        Table = yf.download(self.port_tics, period='1y', parse_dates=True)['Adj Close']
        PT = pd.DataFrame(Table.iloc[1:])
        # PT = PT.fillna(0.0, axis='columns')

        tickers = list(PT.columns)
        returns = PT.pct_change()
        returns.fillna(0.0, inplace=True)
        returns.dropna(inplace=True)
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_portfolios = 5000
        risk_free_rate = 0.0178


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


        def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
            Returns = np.sum(mean_returns*weights ) *252
            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            return std, Returns
        

        def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
            results = np.zeros((3,num_portfolios))
            weights_record = []
            for i in range(num_portfolios):
                weights = np.random.random(len(tickers))
                weights /= np.sum(weights)
                weights_record.append(weights)
                portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
                results[0,i] = portfolio_std_dev
                results[1,i] = portfolio_return
                results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
            return results, weights_record


        def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
            results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
            max_sharpe_idx = np.argmax(results[2])
            sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
            max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=PT.columns,columns=['allocation'])
            max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
            max_sharpe_allocation = max_sharpe_allocation.T
            min_vol_idx = np.argmin(results[0])
            sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
            min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=PT.columns,columns=['allocation'])
            min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
            min_vol_allocation = min_vol_allocation.T

            st.write("-"*80)
            st.subheader(f"{self.saveName0}: Maximum Sharpe Ratio Portfolio Allocation\n")
            st.write(f"* Annualised Return: {round(rp,2)}")
            st.write(f"* Annualised Volatility: {round(sdp,2)}")
            st.dataframe(max_sharpe_allocation.T)
            st.subheader(f"{self.saveName0}: Minimum Volatility Portfolio Allocation\n")
            st.write(f"Annualised Return: {round(rp_min,2)}")
            st.write(f"Annualised Volatility: {round(sdp_min,2)}")
            st.write("-"*80)
            st.dataframe(min_vol_allocation.T)

        # PLOT * * * 
            fig, ax = plt.subplots()
            plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
            plt.colorbar()
            plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
            plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
            plt.title(f'{self.saveName0} - Simulated Portfolio Optimization based on Efficient Frontier')
            plt.xlabel('annualised volatility')
            plt.ylabel('annualised returns')
            plt.legend(labelspacing=0.8, loc='best')
            plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            return rp, sdp, rp_min, sdp_min


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


        def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
            p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
            return -(p_ret - risk_free_rate) / p_var


        def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
            num_assets = len(mean_returns)
            args = (mean_returns, cov_matrix, risk_free_rate)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bound = (0.0,1.0)
            bounds = tuple(bound for asset in range(num_assets))
            result = sco.minimize(
                neg_sharpe_ratio, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
            return result


        def portfolio_volatility(weights, mean_returns, cov_matrix):
            return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]


        def min_variance(mean_returns, cov_matrix):
            num_assets = len(mean_returns)
            args = (mean_returns, cov_matrix)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bound = (0.0,1.0)
            bounds = tuple(bound for asset in range(num_assets))
            result = sco.minimize(
                portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
            return result


        def efficient_return(mean_returns, cov_matrix, target):
            num_assets = len(mean_returns)
            args = (mean_returns, cov_matrix)
            def portfolio_return(weights):
                return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]
            constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0,1) for asset in range(num_assets))
            result = sco.minimize(
                portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
            return result


        def efficient_frontier(mean_returns, cov_matrix, returns_range):
            efficients = []
            for ret in returns_range:
                efficients.append(efficient_return(mean_returns, cov_matrix, ret))
            return efficients


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


        def display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
            results, _ = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
            max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
            sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
            max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=PT.columns,columns=['allocation'])
            max_sharpe_allocation['allocation'] = [round(i*100,2)for i in max_sharpe_allocation.allocation]
            max_sharpe_allocation = max_sharpe_allocation.T
            max_sharpe_allocation
            min_vol = min_variance(mean_returns, cov_matrix)
            sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
            min_vol_allocation = pd.DataFrame(min_vol.x,index=PT.columns,columns=['allocation'])
            min_vol_allocation['allocation'] = [round(i*100,2)for i in min_vol_allocation.allocation]
            min_vol_allocation = min_vol_allocation.T

            st.write("-"*80)
            st.subheader(f"{self.saveName0}: Maximum Sharpe Ratio Portfolio Allocation\n")
            st.write(f"* Annualised Return: {round(rp,2)}")
            st.write(f"* Annualised Volatility: {round(sdp,2)}")
            st.dataframe(max_sharpe_allocation.T)
            st.subheader(f"{self.saveName0}: Minimum Volatility Portfolio Allocation\n")
            st.write(f"* Annualised Return: {round(rp_min,2)}")
            st.write(f"* Annualised Volatility: {round(sdp_min,2)}")
            st.write("-"*80)
            st.dataframe(min_vol_allocation.T)            

        # PLOT * * * 
            fig, ax = plt.subplots()
            plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
            plt.colorbar()
            plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
            plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
            target = np.linspace(rp_min, 0.32, 50)
            efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
            plt.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
            plt.title(f'{self.saveName0} - Calculated Portfolio Optimization based on Efficient Frontier')
            plt.xlabel('annualised volatility')
            plt.ylabel('annualised returns')
            plt.legend(labelspacing=0.8, loc='best')
            plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            return rp, sdp, rp_min, sdp_min


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


        def display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate):
            max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
            sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
            max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=PT.columns, columns=['allocation'])
            max_sharpe_allocation['allocation'] = [round(i*100,2)for i in max_sharpe_allocation.allocation]
            max_sharpe_allocation = max_sharpe_allocation.T
            max_sharpe_allocation
            min_vol = min_variance(mean_returns, cov_matrix)
            sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
            min_vol_allocation = pd.DataFrame(min_vol.x,index=PT.columns,columns=['allocation'])
            min_vol_allocation['allocation'] = [round(i*100,2)for i in min_vol_allocation.allocation]
            min_vol_allocation = min_vol_allocation.T
            an_vol = np.std(PT.pct_change()) * np.sqrt(252)
            an_rt = mean_returns * 252

            st.write("-"*80)
            st.subheader(f"{self.saveName0}: Maximum Sharpe Ratio Portfolio Allocation\n")
            st.write(f"* Annualised Return: {round(rp,2)}")
            st.write(f"* Annualised Volatility: {round(sdp,2)}")
            st.dataframe(max_sharpe_allocation.T)
            st.subheader(f"{self.saveName0}: Minimum Volatility Portfolio Allocation\n")
            st.write(f"* Annualised Return: {round(rp_min,2)}")
            st.write(f"* Annualised Volatility: {round(sdp_min,2)}")
            st.dataframe(min_vol_allocation.T)
            st.write("-"*80)
            st.header("Individual Stock Returns and Volatility\n")
            for i, txt in enumerate(PT.columns):
                st.text(f"{txt}: annuaised return {round(an_rt[i],2)} - annualised volatility: {round(an_vol[i],2)}")
            st.write("-"*80)

        # PLOT * * * 
            fig, ax = plt.subplots()
            ax.scatter(an_vol,an_rt,marker='o',s=200)
            for i, txt in enumerate(PT.columns):
                ax.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')
            ax.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
            ax.scatter(sdp_min,rp_min,marker='*', color='g',s=500, label='Minimum volatility')
            target = np.linspace(rp_min, 0.34, 50)
            efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
            ax.plot([p['fun'] for p in efficient_portfolios], target, linestyle='-.', color='black', label='efficient frontier')
            ax.set_title(f'{self.saveName0} - Portfolio Optimization with Individual Stocks')
            ax.set_xlabel('annualised volatility')
            ax.set_ylabel('annualised returns')
            ax.legend(labelspacing=0.8, loc='best')
            plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            return rp, sdp, rp_min, sdp_min


        rpA, sdpA, rp_minA, sdp_minA = display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
        rpB, sdpB, rp_minB, sdp_minB = display_calculated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)
        rp, sdp, rp_min, sdp_min = display_ef_with_selected(mean_returns, cov_matrix, risk_free_rate)

        return


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


if __name__ == '__main__':
    my_positions = pd.read_pickle("/home/gordon/gdp/project_active/Forecasting_For_Friends/tickers/jayci_ticker_lst.pkl")
    The_Portfolio_Optimizer(my_positions, 'my_positions').optimize()


# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# #*     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *     *
# #* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *