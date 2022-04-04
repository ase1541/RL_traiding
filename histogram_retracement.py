# Librerias utilizadas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
pd.set_option('display.float_format',  '{:,.2f}'.format)

class histogram_retracement() :
    def __init__(self, stock, dataframe, k_entry, k_exit, EMA_days_12, EMA_days_26, STD_rw, MXMN_rw):
        self.stock = stock
        self.dataframe = dataframe
        self.k_entry = k_entry
        self.k_exit = k_exit
        self.EMA_days_12 = EMA_days_12
        self.EMA_days_26 = EMA_days_26
        self.STD_rw = STD_rw
        self.MXMN_rw = MXMN_rw


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
        name = str(self.stock)
        self.dataframe[f'12EMA_{name}'] = self.dataframe[f'{self.stock}'].ewm(span=self.EMA_days_12,
                                                                                    adjust=False, axis=0).mean()
        self.dataframe[f'26EMA_{name}'] = self.dataframe[f'{self.stock}'].ewm(span=self.EMA_days_26, adjust=False, axis=0).mean()
        self.dataframe[f'MACD_{name}'] = self.dataframe[f'12EMA_{self.stock}']-self.dataframe[f'26EMA_{self.stock}']
        self.dataframe[f'Senal_MACD_{name}']=self.dataframe[f'MACD_{self.stock}'].ewm(span=9, adjust=False, axis=0).mean()
        self.dataframe[f'Histograma_MACD_{name}']=self.dataframe[f'MACD_{self.stock}']-self.dataframe[f'Senal_MACD_{self.stock}']
        #Troughs and peaks are needed for the 2nd condition of long signals and short signals
        self.dataframe[f'Trough_{name}']=self.dataframe[f'Histograma_MACD_{self.stock}'].rolling(self.MXMN_rw).min()
        self.dataframe[f'Peak_{name}']=self.dataframe[f'Histograma_MACD_{self.stock}'].rolling(self.MXMN_rw).max()
        #get_max_min(self.dataframe,self.stock)
        #Standard deviation is needed for the 1st condition of long signals and short signals
        self.dataframe[f"Daily_returns_{self.stock}"]=self.dataframe[self.stock].pct_change()
        self.dataframe[f'20STD_{name}']=self.dataframe[f"Daily_returns_{self.stock}"].rolling(self.STD_rw).std()
        self.sell_buy_function()
        self.dataframe.fillna(0, inplace=True) #We eliminate the NaN

        return 0



    def sell_buy_function(self):
        """This function creates for a given self.dataframe, a column of daily returns, sell and buy signals according to
        histogram retracement signals, position and SL, TP"""
        name = str(self.stock)
        rows, columns = self.dataframe.shape
        # series to be analized
        position = pd.Series(data=np.zeros(shape=rows), name="Position")
        sell_signal = pd.Series(data=np.zeros(shape=rows), name="Sell_Signal")
        buy_signal = pd.Series(data=np.zeros(shape=rows), name="Buy_Signal")
        #stop_loss = pd.Series(data=np.zeros(shape=rows), name="stop_loss")
        #take_profit = pd.Series(data=np.zeros(shape=rows), name="take_profit")
        His_entry_long = pd.Series(data=np.zeros(shape=rows), name="Histogram entry long")
        His_entry_short = pd.Series(data=np.zeros(shape=rows), name="Histogram entry short")
        # condiciones
        condicion_sell = np.where(((self.dataframe[f'Histograma_MACD_{name}']) > 0) & (
                    (self.dataframe[f'Histograma_MACD_{name}']) < (self.k_entry * self.dataframe[f'Peak_{name}'])) & (
                                              (self.dataframe[f'Histograma_MACD_{name}']) > (
                                                  0.5 * self.dataframe[f'20STD_{name}'])), 1, 0)
        condicion_buy = np.where(((self.dataframe[f'Histograma_MACD_{name}']) < 0) & (
                    (self.dataframe[f'Histograma_MACD_{name}']) > (self.k_entry * self.dataframe[f'Trough_{name}'])) & (
                                             (self.dataframe[f'Histograma_MACD_{name}']) < (
                                                 -0.5 * self.dataframe[f'20STD_{name}'])), 1, 0)

        for row in range(1, rows):
            # if we have no position
            if position[row - 1] == 0:
                if condicion_sell[row - 1]:
                    sell_signal[row] = 1
                    position[row] = -1
                    # stop_loss[row]=self.dataframe[self.stock][row]*SL["S"]
                    # take_profit[row]=self.dataframe[self.stock][row]*TP["S"]
                    His_entry_short[row] = self.dataframe[f'Histograma_MACD_{name}'][row]
                elif condicion_buy[row - 1]:
                    buy_signal[row] = 1
                    position[row] = 1
                    # stop_loss[row]=self.dataframe[self.stock][row]*SL["L"]
                    # take_profit[row]=self.dataframe[name][row]*TP["L"]
                    His_entry_long[row] = self.dataframe[f'Histograma_MACD_{name}'][row]
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
                if (self.dataframe[f'Histograma_MACD_{name}'][row] > self.k_exit * His_entry_long[row - 1]):
                    position[row] = 1
                if (self.dataframe[f'Histograma_MACD_{name}'][row] < self.k_exit * His_entry_long[row - 1]):
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
                if (self.dataframe[f'Histograma_MACD_{name}'][row] < self.k_exit * His_entry_long[row - 1]):
                    position[row] = -1
                if (self.dataframe[f'Histograma_MACD_{name}'][row] > self.k_exit * His_entry_long[row - 1]):
                    position[row] = 0

        self.dataframe[f"Sell_signal_{name}"] = sell_signal.values
        self.dataframe[f"Buy_signal_{name}"] = buy_signal.values
        self.dataframe[f"Position_{name}"] = position.values
        self.dataframe[f"His_entry_long_{name}"] = His_entry_long.values
        self.dataframe[f"His_entry_short_{name}"] = His_entry_short.values
        # self.dataframe[f"SL_{self.stock}"]=stop_loss.values
        # self.dataframe[f"TP_{self.stock}"]=take_profit.values

        return 0



    def get_max_min(self):
        """ Creates and appends two columns to the self.dataframe that have the maximum and minimum value of the self.stock.
        it is build such that it gets the max and min of every time the funcion Histogram_MACD crosses zero. """

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
        w = pd.DataFrame(index=["Total"])
        index1.append(w.index)
        returns_ = pd.DataFrame(columns=columns, index=index1, data=0)
        for row in index0:
            try:
                # Short
                returns_.loc[row, "Yearly Return Short"] = \
                self.dataframe[f"Daily_returns_{self.stock}"].loc[self.dataframe[f"Position_{self.stock}"] == -1][str(row)].sum() * -100
                returns_.loc[row, "Volatility Short"] = \
                self.dataframe[f"Daily_returns_{self.stock}"].loc[self.dataframe[f"Position_{self.stock}"] == -1][str(row)].std() * -100
                returns_.loc[row, "Sharp Ratio Short"] = (
                            returns_.loc[row, "Yearly Return Short"] / returns_.loc[row, "Volatility Short"])
            except:
                pass
            # Long
            try:
                returns_.loc[row, "Yearly Return Long"] = \
                self.dataframe[f"Daily_returns_{self.stock}"].loc[self.dataframe[f"Position_{self.stock}"] == 1][str(row)].sum() * 100
                returns_.loc[row, "Volatility Long"] = \
                self.dataframe[f"Daily_returns_{self.stock}"].loc[self.dataframe[f"Position_{self.stock}"] == 1][str(row)].std() * 100
                returns_.loc[row, "Sharp Ratio Long"] = (
                            returns_.loc[row, "Yearly Return Long"] / returns_.loc[row, "Volatility Long"])
                returns_.fillna(0, inplace=True)  # We eliminate the NaN
            except:
                pass
        # as the row total is not an iterable, but a result of the previous, we put it out of the loop  
        returns_.loc["Total", :] = [returns_["Yearly Return Short"].sum(), returns_["Volatility Short"].sum(),
                                    returns_["Sharp Ratio Short"].sum(), returns_["Yearly Return Long"].sum(),
                                    returns_["Volatility Long"].sum(), returns_["Sharp Ratio Long"].sum()]
        Total_Return_Short = returns_.loc["Total", "Yearly Return Short"]
        Total_Return_Long = returns_.loc["Total", "Yearly Return Long"]
        Total_sum = Total_Return_Short + Total_Return_Long
        return returns_, Total_Return_Short, Total_Return_Long, Total_sum



    # Plot graficas con Histograma_MACD_

    def plot_signals(self):
        
        fig, ax = plt.subplots(figsize=(30, 9))
        # fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Trading strategy')
        ax.set_title('Histogram retracement', fontsize=18, fontweight='bold')
        name=str(self.stock)
        self.dataframe[f"Histograma_MACD_{name}"].plot(ax=ax)  # Histogram
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
        ax1.set_ylabel(f'{name} Price')
        ax1.set_title(f'{name} Stock Price', fontsize=18, fontweight='bold')
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
        return fig, fig1
