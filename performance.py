
## We perform the test of the best algorithm performance in terms of sharpe ratio for a rolling window
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from train_RL import val_data
import pyfolio

def get_daily_return(df):
    df['daily_return']=df.account_value.pct_change(1)
    #df=df.dropna()
    print('Sharpe: ',(252**0.5)*df['daily_return'].mean()/ df['daily_return'].std())
    return df

def backtest_strat(df):
    strategy_ret= df.copy()
    strategy_ret['Date'] = pd.to_datetime(strategy_ret['Date'])
    strategy_ret.set_index('Date', drop = False, inplace = True)
    strategy_ret.index = strategy_ret.index.tz_localize('UTC')
    del strategy_ret['Date']
    ts = pd.Series(strategy_ret['daily_return'].values, index=strategy_ret.index)
    return ts

benchmark = val_data
benchmark["Date"] = benchmark.index
balance_data = pd.read_csv("results/account_value_val_PPO.csv")

balance_data['Date'] = pd.to_datetime(balance_data['Date'])
#benchmark['Date'] = pd.to_datetime(benchmark['Date'])

balance_data['Daily_returns']=balance_data["Balance"].pct_change(1)
benchmark['Daily_returns'] = benchmark["Close"].pct_change(1)

balance_data.drop(columns=["Unnamed: 0","Balance"], inplace=True, axis=0)
benchmark.drop(columns=["Open","High", "Close", "Low", "Hist_Retrace","Row","Volume"], inplace=True, axis=0)
balance_data.set_index('Date', drop = True, inplace = True)

benchmark.set_index('Date', drop = True, inplace = True)

#Doesn?t work yet
pyfolio.create_returns_tear_sheet(returns = balance_data["Daily_returns"], benchmark_rets= benchmark["Daily_returns"],set_context=False)
