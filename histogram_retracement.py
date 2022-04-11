# Librerias utilizadas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

pd.set_option('display.float_format',  '{:,.2f}'.format)

class histogram_retracement() :
    def __init__(self, stock, dataframe, k_entry, k_exit, EMA_days_12, EMA_days_26, STD_rw, MXMN_rw):
        self.stock = stock
        self.name = str(self.stock)
        self.dataframe = dataframe
        self.k_entry = k_entry
        self.k_exit = k_exit
        self.EMA_days_12 = EMA_days_12
        self.EMA_days_26 = EMA_days_26
        self.STD_rw = STD_rw
        self.MXMN_rw = MXMN_rw

    def count_trades(self):
        trades = {"Type of trade":[], "start":[],"end":[], "returns":[], "sharp":[], "Nº of days":[]}
        rows = self.dataframe.shape[0]
        Position = self.dataframe[f'Position_{self.stock}']
        inicio=0
        final=0
        for row in range(1,rows):
            if ((Position[row] == 1) or (Position[row] == -1)) & (Position[row-1] == 0):
                if (Position[row] == 1):
                    inicio=row
                    trades["start"].append(Position.index[row])
                    trades["Type of trade"].append("Long")
                if (Position[row] == -1):
                    inicio=row
                    trades["start"].append(Position.index[row])
                    trades["Type of trade"].append("Short")
            if ((Position[row] == 0) & ((Position[row - 1] == 1) or (Position[row - 1] == -1))):
                    if (Position[row-1] == 1):
                        final=row
                        trades["end"].append(Position.index[row])
                        trades["Nº of days"].append(final-inicio)
                        return_long = self.dataframe[f"Daily_returns_{self.stock}"].iloc[inicio:final]
                        trades["returns"].append(return_long.sum()*100)
                        trades["sharp"].append((252**0.5)*(return_long.mean()/return_long.std()))
                        if return_long.std() == np.nan:
                            trades["sharp"].append(0)
                        else:
                            trades["sharp"].append((252 ** 0.5) * (return_long.mean() / return_long.std()))
                        inicio = 0
                        final = 0
                    if (Position[row-1] == -1):
                        final = row
                        trades["end"].append(Position.index[row])
                        trades["Nº of days"].append(final - inicio)
                        return_short = self.dataframe[f"Daily_returns_{self.stock}"].iloc[inicio:final]
                        trades["returns"].append(return_short.sum()*100)
                        if return_short.std() == np.nan:
                            trades["sharp"].append(0)
                        else:
                            trades["sharp"].append((252 ** 0.5) * (return_short.mean() / return_short.std()))
                        inicio = 0
                        final = 0
        #trades["sharp"]= [0 for i in trades["sharp"] if i == np.nan]
        newlist = [x for x in trades["sharp"] if math.isnan(x) == False]

        resumen=f"""Se han realizado {len(trades["Type of trade"])} operaciones de trading
        - {trades["Nº of days"].count(1)} Operaciones duran solo un día
        - {trades["Type of trade"].count("Long")} Son Short Selling
        - {trades["Type of trade"].count("Short")} Son Long buying
El sharp medio es de: {sum(newlist)/len(trades["sharp"])} 
El retorno medio es de: {sum(trades["returns"])/len(trades["returns"])} %
La duración media de las operaciones es de: {sum(trades["Nº of days"])/len(trades["Nº of days"])} Días
                    """
        return print(resumen), trades


    def signal_construction (self):
        """Signal construction
    •	MACD = 12-day exponential moving average of close – 26-day exponential moving average of close
    •	MACD Signal = 9-day exponential moving average of MACD
    •	MACD Histogram = MACD – MACD Signal
    •	Buy (long) signal when the histogram retraces x percent of its prior trough when below zero.
    •	Sell (short) signal if the histogram retraces x percent of its prior peak when above zero.

    To avoid whipsaws, we require the histogram to reach some minimum threshold above zero before taking short entry
    signals, and some minimum threshold below zero before taking long entries.

    Short entering conditions
    •	It requires the histogram to exceed one-half of the standard deviation of price changes over the past 20 days
    before taking short signals.
    •	Long signals are entered if the histogram retraces 25 percent of its minimum value since its last
    cross below zero.

    Long entering conditions
    •	We require the histogram to be less than negative onehalf times the standard deviation of price changes over
    the past 20 days before taking long signals.
    •	Short signals are entered if the histogram retraces 25 percent of its maximum value since its last cross
    above zero.
    """

        EMA_12 = self.dataframe[f'{self.stock}'].ewm(span=self.EMA_days_12, adjust=False, axis=0).mean()
        EMA_26 = self.dataframe[f'{self.stock}'].ewm(span=self.EMA_days_26, adjust=False, axis=0).mean()
        MACD = EMA_12-EMA_26
        Senal_MACD =MACD.ewm(span=9, adjust=False, axis=0).mean()
        self.dataframe[f'Histograma_MACD_{self.name}']=MACD-Senal_MACD
        #Troughs and peaks are needed for the 2nd condition of long signals and short signals

        #get_max_min(self.dataframe,self.stock)
        #Standard deviation is needed for the 1st condition of long signals and short signals
        self.dataframe[f"Daily_returns_{self.stock}"]=self.dataframe[self.stock].pct_change(1)
        self.sell_buy_function()
        self.dataframe.fillna(0, inplace=True) #We eliminate the NaN

        return 0



    def sell_buy_function(self):
        """This function creates for a given self.dataframe, a column of daily returns, sell and buy signals according to
        histogram retracement signals, position and SL, TP"""

        rows, columns = self.dataframe.shape

        # Position information
        position = pd.Series(data=np.zeros(shape=rows), name="Position")
        sell_signal = pd.Series(data=np.zeros(shape=rows), name="Sell_Signal")
        buy_signal = pd.Series(data=np.zeros(shape=rows), name="Buy_Signal")
        #stop_loss = pd.Series(data=np.zeros(shape=rows), name="stop_loss")
        #take_profit = pd.Series(data=np.zeros(shape=rows), name="take_profit")
        #Columns for conditions
        His_entry_long = pd.Series(data=np.zeros(shape=rows), name="Histogram entry long")
        His_entry_short = pd.Series(data=np.zeros(shape=rows), name="Histogram entry short")
        Trough = self.dataframe[f'Histograma_MACD_{self.stock}'].rolling(self.MXMN_rw).min()
        Peak = self.dataframe[f'Histograma_MACD_{self.stock}'].rolling(self.MXMN_rw).max()
        STD_20=self.dataframe[f"Daily_returns_{self.stock}"].rolling(self.STD_rw).std()
        #Conditions
        condicion_sell = np.where(((self.dataframe[f'Histograma_MACD_{self.name}']) > 0) & (
                    (self.dataframe[f'Histograma_MACD_{self.name}']) < (self.k_entry * Peak)) & (
                                              (self.dataframe[f'Histograma_MACD_{self.name}']) > (
                                                  0.5 * STD_20)), 1, 0)
        condicion_buy = np.where(((self.dataframe[f'Histograma_MACD_{self.name}']) < 0) & (
                    (self.dataframe[f'Histograma_MACD_{self.name}']) > (self.k_entry * Trough)) & (
                                             (self.dataframe[f'Histograma_MACD_{self.name}']) < (
                                                 -0.5 * STD_20)), 1, 0)

        for row in range(1, rows):
            # if we have no position
            if position[row - 1] == 0:
                if condicion_sell[row - 1]:
                    sell_signal[row] = 1
                    position[row] = -1
                    # stop_loss[row]=self.dataframe[self.stock][row]*SL["S"]
                    # take_profit[row]=self.dataframe[self.stock][row]*TP["S"]
                    His_entry_short[row] = self.dataframe[f'Histograma_MACD_{self.name}'][row]
                elif condicion_buy[row - 1]:
                    buy_signal[row] = 1
                    position[row] = 1
                    # stop_loss[row]=self.dataframe[self.stock][row]*SL["L"]
                    # take_profit[row]=self.dataframe[name][row]*TP["L"]
                    His_entry_long[row] = self.dataframe[f'Histograma_MACD_{self.name}'][row]
                else:
                    sell_signal[row] = 0
                    buy_signal[row] = 0
                    position[row] = 0
            # if we are long
            elif position[row - 1] == 1:
                # stop_loss[row]=stop_loss[row-1]
                # take_profit[row]=take_profit[row-1]
                His_entry_long[row] = His_entry_long[row - 1]
                """if (self.dataframe[self.stock][row] > stop_loss[row-1]) or (self.dataframe[self.stock][row] < take_profit[row-1]):
                    position[row]= 1
                if (self.dataframe[self.stock][row] < stop_loss[row-1]) or (self.dataframe[self.stock][row] > take_profit[row-1]):
                    position[row]= 0"""
                if (self.dataframe[f'Histograma_MACD_{self.name}'][row] > self.k_exit * His_entry_long[row - 1]):
                    position[row] = 1
                if (self.dataframe[f'Histograma_MACD_{self.name}'][row] < self.k_exit * His_entry_long[row - 1]):
                    position[row] = 0

            # if we are short
            elif position[row - 1] == -1:
                # stop_loss[row]=stop_loss[row-1]
                # take_profit[row]=take_profit[row-1]
                His_entry_short[row] = His_entry_short[row - 1]
                """if (self.dataframe[self.stock][row] < stop_loss[row-1]) or (self.dataframe[self.stock][row] > take_profit[row-1]):
                    position[row]= -1
                if (self.dataframe[self.stock][row] > stop_loss[row-1]) or (self.dataframe[self.stock][row] < take_profit[row-1]):
                    position[row]= 0"""
                if (self.dataframe[f'Histograma_MACD_{self.name}'][row] < self.k_exit * His_entry_long[row - 1]):
                    position[row] = -1
                if (self.dataframe[f'Histograma_MACD_{self.name}'][row] > self.k_exit * His_entry_long[row - 1]):
                    position[row] = 0

        self.dataframe[f"Sell_signal_{self.name}"] = sell_signal.values
        self.dataframe[f"Buy_signal_{self.name}"] = buy_signal.values
        self.dataframe[f"Position_{self.name}"] = position.values
        #self.dataframe[f"His_entry_long_{name}"] = His_entry_long.values
        #self.dataframe[f"His_entry_short_{name}"] = His_entry_short.values
        # self.dataframe[f"SL_{self.stock}"]=stop_loss.values
        # self.dataframe[f"TP_{self.stock}"]=take_profit.values

        return 0



    def get_max_min(self):
        """ Creates and appends two columns to the self.dataframe that have the maximum and minimum value of the self.stock.
        it is built to get the max and min in between two Histogram_MACD crosses zero. """

        maxs = []  # list to contain max value
        pos_maxs = []  # position max
        mins = []  # list to contain min value
        pos_mins = []  # position min
        self.stock_min = []
        self.stock_max = []
        interval_ini = 0  # marker of the start of a non zero interval
        interval_fin = 0  # marker of the end of a non zero interval
        binario_positivos = np.where((self.dataframe[f'Histograma_MACD_{self.stock}'] > 0), 1, 0)
        binario_negativos = np.where((self.dataframe[f'Histograma_MACD_{self.stock}'] < 0), 1, 0)
        Peaks = pd.Series(data=np.zeros(shape=binario_positivos.shape[0]), name="Peaks")
        Troughs = pd.Series(data=np.zeros(shape=binario_negativos.shape[0]), name="Troughs")
        for i in range(1, binario_positivos.shape[0]):
            if (binario_positivos[i] == 1) & (binario_positivos[i - 1] == 0):
                interval_ini = i

            if (binario_positivos[i] == 0) & (binario_positivos[i - 1] == 1):
                interval_fin = i

            if (interval_fin != 0) & (interval_ini != 0):
                y = self.dataframe[f'Histograma_MACD_{self.stock}'][interval_ini:interval_fin].max()
                x = self.dataframe.loc[self.dataframe[f'Histograma_MACD_{self.stock}'] == y].index
                # maxs.append(y)
                # pos_maxs.append(x)
                # self.stock_max.append(close_data[self.stock][x])
                Peaks[interval_ini:interval_fin] = y
                interval_fin = 0
                interval_ini = 0
        interval_ini = 0  # marker of the start of a non zero interval
        interval_fin = 0  # marker of the end of a non zero interval
        for i in range(1, binario_negativos.shape[0]):
            if (binario_negativos[i] == 1) & (binario_negativos[i - 1] == 0):
                interval_ini = i

            if (binario_negativos[i] == 0) & (binario_negativos[i - 1] == 1):
                interval_fin = i

            if (interval_fin != 0) & (interval_ini != 0):
                y1 = self.dataframe[f'Histograma_MACD_{self.stock}'][interval_ini:interval_fin].min()
                x1 = self.dataframe.loc[self.dataframe[f'Histograma_MACD_{self.stock}'] == y1].index
                # mins.append(y1)
                # pos_mins.append(x1)
                # self.stock_min.append(close_data[self.stock][x1])
                Troughs[interval_ini:interval_fin] = y1
                interval_fin = 0
                interval_ini = 0
        self.dataframe[f'Peak_{self.stock}'] = Peaks.values
        self.dataframe[f'Trough_{self.stock}'] = Troughs.values
        return 0  # maxs, pos_maxs, mins, pos_mins,self.stock_max, self.stock_min

    def get_returns(self):
        """This function creates a pandas table with the info about annual returns, volatility and 
        sharpe ratio anually for a self.stock, and also delivers the total sum per column in the last row """
        # We create the pandas table to save the data
        columns = ["Yearly Return Short", "Volatility Short", "Sharp Ratio Short", "Yearly Return Long",
                   "Volatility Long", "Sharp Ratio Long"]
        index0 = self.dataframe.resample('Y').sum().index.year
        index1 = self.dataframe.resample('Y').sum().index.year
        # We add a row called total to calculate the total value of the rest
        w = pd.DataFrame(index=["Promedio"])
        index1.append(w.index)
        returns_ = pd.DataFrame(columns=columns, index=index1, data=0)
        for row in index0:
            # Short
            try:
                short_returns=self.dataframe[f"Daily_returns_{self.stock}"].loc[self.dataframe[f"Position_{self.stock}"] == -1][str(row)]
                returns_.loc[row, "Yearly Return Short"] = short_returns.sum() * -100
                returns_.loc[row, "Volatility Short"] = (252**0.5)*short_returns.std() * 100
                returns_.loc[row, "Sharp Ratio Short"] = ((12**0.5)* short_returns.mean() / short_returns.std())
            except:
                pass
            # Long
            try:
                long_returns=self.dataframe[f"Daily_returns_{self.stock}"].loc[self.dataframe[f"Position_{self.stock}"] == 1][str(row)]
                returns_.loc[row, "Yearly Return Long"] = long_returns.sum() * 100
                returns_.loc[row, "Volatility Long"] = (252**0.5)*long_returns.std() * 100
                returns_.loc[row, "Sharp Ratio Long"] = ((252**0.5)*long_returns.mean()/long_returns.std())
                returns_.fillna(0, inplace=True)  # We eliminate the NaN
            except:
                pass
        # as the row total is not an iterable, but a result of the previous, we put it out of the loop  
        returns_.loc["Promedio", :] = [returns_["Yearly Return Short"].mean(), returns_["Volatility Short"].mean(),
                                    returns_["Sharp Ratio Short"].mean(), returns_["Yearly Return Long"].mean(),
                                    returns_["Volatility Long"].mean(), returns_["Sharp Ratio Long"].mean()]
        Avg_Return_Short = returns_.loc["Promedio", "Yearly Return Short"]
        Avg_Return_Long = returns_.loc["Promedio", "Yearly Return Long"]
        Avg_sum = Avg_Return_Short + Avg_Return_Long
        return returns_, Avg_Return_Short, Avg_Return_Long, Avg_sum



    # Plot graficas con Histograma_MACD_

    def plot_signals(self):
        
        fig, ax = plt.subplots(figsize=(30, 9))
        # fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Trading strategy')
        ax.set_title('Histogram retracement', fontsize=18, fontweight='bold')
        self.dataframe[f"Histograma_MACD_{self.name}"].plot(ax=ax)  # Histogram
        ax.plot(self.dataframe.index, np.zeros(self.dataframe[self.stock].shape[0]))  # line of zeros
        # señales de compra
        ax.plot(self.dataframe.loc[self.dataframe[f"Sell_signal_{self.stock}"] == 1].index,
                self.dataframe[f'Histograma_MACD_{self.stock}'][self.dataframe[f"Sell_signal_{self.stock}"] == 1], 'v', markersize=10,
                color='r', label="Selling signals")
        # señales de venta
        ax.plot(self.dataframe.loc[self.dataframe[f"Buy_signal_{self.stock}"] == 1].index,
                self.dataframe[f'Histograma_MACD_{self.stock}'][self.dataframe[f"Buy_signal_{self.stock}"] == 1], '^', markersize=10,
                color='g', label="Buying signals")
        ax.legend()

        fig1 = plt.figure()  # an empty figure with no Axes
        fig1, ax1 = plt.subplots(figsize=(30, 9))
        ax1.set_xlabel('Time')
        ax1.set_ylabel(f'{self.name} Price')
        ax1.set_title(f'{self.name} Stock Price', fontsize=18, fontweight='bold')
        ax1.plot(self.dataframe[self.stock].index, self.dataframe[self.stock], label="self.stock price")
        # señales de venta
        ax1.plot(self.dataframe.loc[self.dataframe[f"Sell_signal_{self.stock}"] == 1].index,
                 self.dataframe[self.stock][self.dataframe[f"Sell_signal_{self.stock}"] == 1], 'v', markersize=10, color='r',
                 label="Selling signals")
        # señales de compra
        ax1.plot(self.dataframe.loc[self.dataframe[f"Buy_signal_{self.stock}"] == 1].index,
                 self.dataframe[self.stock][self.dataframe[f"Buy_signal_{self.stock}"] == 1], '^', markersize=10, color='g',
                 label="Buying signals")
        ax1.legend()
        plt.show()
        return fig, fig1
