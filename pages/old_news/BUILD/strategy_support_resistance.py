import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import yfinance as yf
plt.rcParams['figure.figsize'] = [13, 5.5]
plt.rcParams['figure.dpi'] = 150

class Support_Resistance(object):
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.SRC_DATA_FILENAME = '/home/gordon/gdp/code/design/old_projects/one/files/raw/'+self.ticker+'_strategy_support_resist.pkl'
    
        try:
            goog_data2 = pd.read_pickle(self.SRC_DATA_FILENAME)
        except FileNotFoundError:
            goog_data2 = yf.download(self.ticker, start=self.start_date, end=self.end_date, parse_dates=True)
            goog_data2.to_pickle(self.SRC_DATA_FILENAME)

        goog_data=goog_data2.tail(620)
        self.lows=goog_data['Low']
        self.highs=goog_data['High']
        h = self.highs.index.where(self.highs.values >= self.highs.max())
        h = h.dropna()
        l = self.lows.index.where(self.lows.values <= self.lows.min())
        l = l.dropna()

        fig = plt.figure(figsize=(13,6))
        ax1 = fig.add_subplot(111, ylabel='Google price in $')
        goog_data['Adj Close'].plot(ax=ax1, color='k', lw=2., label=self.ticker + ' price')
        self.highs.plot(ax=ax1, color='c', lw=2., label='highs')
        self.lows.plot(ax=ax1, color='y', lw=2., label='lows')
        plt.hlines(self.highs.head(200).max(),self.lows.index.values[0],self.lows.index.values[-1],linewidth=2, label=f'resistance level = ${round(self.highs.max(),2)}')
        plt.hlines(self.lows.head(200).min(),self.lows.index.values[0],self.lows.index.values[-1],linewidth=2, label=f'support level = ${round(self.lows.min(),2)}')
        for i in self.highs.index.values:
            if i == h:
                plt.axvline(x=i,linestyle=':',linewidth=2, color='g', label='high')
        for i in self.lows.index.values:
            if i == l:
                plt.axvline(x=i,linestyle=':',linewidth=2, color='r', label='low')                
        plt.legend()
        plt.title('Visualization of Support & Resistance - Observation of Highs/Lows to Graph Generation')

        try:
            goog_data = pd.read_pickle(self.SRC_DATA_FILENAME)
            print('File data found...reading GOOG data')
        except FileNotFoundError:
            print('File not found...downloading the GOOG data')
            goog_data = data.DataReader('GOOG', 'yahoo', start_date, end_date)
            goog_data.to_pickle(self.SRC_DATA_FILENAME)

        self.goog_data_signal = pd.DataFrame(index=goog_data.index)
        self.goog_data_signal['price'] = goog_data['Adj Close']

    def trading_support_resistance(self,bin_width=20):
        bin_width=20
        data=self.goog_data_signal
        data['sup_tolerance'] = pd.Series(np.zeros(len(data)))
        data['res_tolerance'] = pd.Series(np.zeros(len(data)))
        data['sup_count'] = pd.Series(np.zeros(len(data)))
        data['res_count'] = pd.Series(np.zeros(len(data)))
        data['support'] = pd.Series(np.zeros(len(data)))
        data['resistance'] = pd.Series(np.zeros(len(data)))
        data['positions'] = pd.Series(np.zeros(len(data)))
        data['signal'] = pd.Series(np.zeros(len(data)))
        in_support=0
        in_resistance=0

        for x in range((bin_width - 1) + bin_width, len(data)):
            data_section = data[x - bin_width:x + 1]
            support_level=min(data_section['price'])
            resistance_level=max(data_section['price'])
            range_level=resistance_level-support_level
            data['resistance'][x]=resistance_level
            data['support'][x]=support_level
            data['sup_tolerance'][x]=support_level + 0.2 * range_level
            data['res_tolerance'][x]=resistance_level - 0.2 * range_level

            if data['price'][x]>=data['res_tolerance'][x] and data['price'][x] <= data['resistance'][x]:
                in_resistance+=1
                data['res_count'][x]=in_resistance
            elif data['price'][x] <= data['sup_tolerance'][x] and data['price'][x] >= data['support'][x]:
                in_support += 1
                data['sup_count'][x] = in_support
            else:
                in_support=0
                in_resistance=0
            if in_resistance>2:
                data['signal'][x]=1
            elif in_support>2:
                data['signal'][x]=0
            else:
                data['signal'][x] = data['signal'][x-1]
        data['positions']=data['signal'].diff()

        fig = plt.figure()
        ax1 = fig.add_subplot(111, ylabel='Google price in $')
        self.goog_data_signal['support'].plot(ax=ax1, color='c', lw=1.5)
        self.goog_data_signal['resistance'].plot(ax=ax1, color='y', lw=1.5)
        self.goog_data_signal['price'].plot(ax=ax1, color='k', lw=1.5)
        plt.hlines(self.highs.head(200).max(),self.lows.index.values[0],self.lows.index.values[-1], color='g', linewidth=2, label=f'resistance level = ${round(self.highs.max(),2)}')
        plt.hlines(self.lows.head(200).min(),self.lows.index.values[0],self.lows.index.values[-1], color='r', linewidth=2, label=f'support level = ${round(self.lows.min(),2)}')        
        ax1.plot(
            self.goog_data_signal.loc[self.goog_data_signal.positions == 1.0].index,
            self.goog_data_signal.price[self.goog_data_signal.positions == 1.0],
            '^', markersize=9, color='g',label='buy')
        ax1.plot(
            self.goog_data_signal.loc[self.goog_data_signal.positions == -1.0].index,
            self.goog_data_signal.price[self.goog_data_signal.positions == -1.0],
            'v', markersize=9, color='r',label='sell')
        plt.title('SUPPORT & RESISTANCE vs PRICE - BUY & SELL SIGNALS')
        plt.legend()
        print(self.goog_data_signal)
        plt.show()


if __name__ == '__main__':
    pass        
    Support_Resistance(ticker='AAPL', start_date='2020-01-01', end_date='2021-01-01').trading_support_resistance()