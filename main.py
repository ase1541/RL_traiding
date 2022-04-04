import investpy
import datetime as dt

import pandas as pd

from histogram_retracement import histogram_retracement

# Define all posible combinations for the backtesting
params = {"k_entry": [0.55, 0.75, 0.85, 0.95], "k_exit": [0.15, 0.55, 0.75, 0.95],
          "EMA_days_12": [3, 5, 10, 12], "EMA_days_26": [20, 26, 30, 50],
          "STD_rollingwindow": [10, 20, 30, 50], "MAXMIN_rollingwindow": [10, 26, 30, 50]}

# Define parameters for the investpy API
first_date = '01/01/2015'
# SL={"L":0.97,"S":1.03}
# TP={"L":1.50, "S":0.95} #stop loss and take profit for exit signals
now = dt.datetime.today()
today = now.strftime('%d/%m/%Y')
seleccion_empresas_IBEX35 = ["ALM",
                             "GRLS",
                             "ITX",
                             "ROVI",
                             "PHMR",
                             "COL",
                             "MRL"]
stock = "ALM"
# get stock data
data = investpy.stocks.get_stock_historical_data(stock, country='spain', from_date=first_date,
                                                                   to_date=today, interval='Daily')
serie = "Close"
dataframe=pd.DataFrame(columns=[stock])
dataframe[stock] = data[serie]

k_entry = params["k_entry"][0]  # percentage of peaks and troughs
k_exit = params["k_exit"][0]
EMA_days_12 = params["EMA_days_12"][0]
EMA_days_26 = params["EMA_days_26"][0]
STD_rw = params["STD_rollingwindow"][0]
MXMN_rw = params["MAXMIN_rollingwindow"][0]

strategy = histogram_retracement(stock, dataframe, k_entry, k_exit, EMA_days_12, EMA_days_26, STD_rw, MXMN_rw)
a=strategy.signal_construction()
strategy.dataframe.info()