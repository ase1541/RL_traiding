import investpy
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from histogram_retracement import histogram_retracement



# Define parameters for the investpy API
first_date = '06/12/2007'
now = dt.datetime.today()
today = now.strftime('%d/%m/%Y')
stock = "ROVI"
# get stock data
data = investpy.stocks.get_stock_historical_data(stock, country='spain', from_date=first_date,
                                                                to_date=today, interval='Daily')
dataframe = data.copy()
dataframe.drop(columns=["Currency"], inplace = True)
dataframe.rename(columns={"Close": stock}, inplace=True)

strategy = histogram_retracement(stock=stock, dataframe=dataframe, k_entry=0.95, k_exit=0.75, EMA_days_12=3, EMA_days_26=50, STD_rw=50, MXMN_rw=10)
strategy.signal_construction()
_ ,trades = strategy.count_trades()
returns, _,_,_ = strategy.get_returns()
#strategy.plot_signals()
#plt.show()
# After performing calculus we drop unnecessary columns
dataframe = dataframe.drop(columns=[f"Daily_returns_{stock}", f"Sell_signal_{stock}", f"Buy_signal_{stock}",
             f"Position_{stock}"])
dataframe = dataframe.rename(columns={stock: "Close", f"Histograma_MACD_{stock}": "Hist_Retrace"})
dataframe["Row"] = np.arange(1,len(dataframe.Close)+1) #makes easier the iteration over rows that have a timestamp form


