import backtrader as bt
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 7]
plt.rcParams['figure.dpi'] = 150
import streamlit as st

from pathlib import Path
from datetime import datetime
today = str(datetime.now())[:10]
savePlot = Path(f'report/portfolio_{today}/IIIII_evaluate')
if not savePlot.exists():
    savePlot.mkdir(parents=True)

class BuyAndHold_Buy(bt.Strategy):
    def start(self):       # set the starting cash
        self.val_start = self.broker.get_cash() 

    def nextstart(self):       # Buy stocks with all the available cash
        size = int(self.val_start / self.data)
        self.buy(size=size)

    def stop(self):       # calculate the actual returns
        self.roi = (self.broker.get_value() / self.val_start) - 1.0
        st.write("\n ROI: %.2f, \nCash: %.2f" % (
          1000.0 * self.roi, self.broker.get_value()))

def setup(stock, start, end):
    data = bt.feeds.YahooFinanceData(
        dataname=stock, 
        fromdate=start,
        todate=end
    )
    return data

def xavier(stock, start=datetime(2021, 1, 1), end=datetime.now()):
      data = setup(stock, start, end)
      cerebro = bt.Cerebro()
      cerebro.adddata((data))
      cerebro.addstrategy(BuyAndHold_Buy, "HODL")
      start_cash=1000.00
      cerebro.broker.setcash(start_cash)
      cerebro.run()
      t_name=str(f'backTrader_technicalIndicator_{stock}.png')
      st.pyplot(
          cerebro.plot(
              tic_name=t_name,plotter=None, numfigs=1, iplot=True, start=None, end=None,tight=True, use=None
              )
      )              
        
    #   plt.savefig(savePlot / f'{stock}_bt_buyHold.png')


if __name__ == '__main__':
      xavier(stock='GM')