import investpy
import datetime as dt
import pandas as pd
from histogram_retracement import histogram_retracement

# Define all posible combinations for the backtesting
params = {"k_entry": [0.55, 0.75, 0.85, 0.95], "k_exit": [0.15, 0.55, 0.75, 0.95],
          "EMA_days_12": [3, 5, 10, 12], "EMA_days_26": [20, 26, 30, 50],
          "STD_rollingwindow": [10, 20, 30, 50], "MAXMIN_rollingwindow": [10, 26, 30, 50]}

# Define parameters for the investpy API
first_date = '01/01/2010'
# SL={"L":0.97,"S":1.03}
# TP={"L":1.50, "S":0.95} #stop loss and take profit for exit signals
now = dt.datetime.today()
today = now.strftime('%d/%m/%Y')
stock = "ROVI"
# get stock data
data = investpy.stocks.get_stock_historical_data(stock, country='spain', from_date=first_date,
                                                                   to_date=today, interval='Daily')
serie = "Close"
dataframe=pd.DataFrame(columns=[stock])
dataframe[stock] = data[serie]

h=0
k_entry = params["k_entry"][h]  # percentage of peaks and troughs
k_exit = params["k_exit"][h]
EMA_days_12 = params["EMA_days_12"][h]
EMA_days_26 = params["EMA_days_26"][h]
STD_rw = params["STD_rollingwindow"][h]
MXMN_rw = params["MAXMIN_rollingwindow"][h]

strategy = histogram_retracement(stock, dataframe, k_entry, k_exit, EMA_days_12, EMA_days_26, STD_rw, MXMN_rw)
strategy.signal_construction()
_ ,trades = strategy.count_trades()
a=strategy.get_returns()


