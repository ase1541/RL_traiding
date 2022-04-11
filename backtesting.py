from main import params, dataframe, stock
from histogram_retracement import histogram_retracement
import itertools
# Define all posible combinations for the backtesting
keys = params.keys()
values = (params[key] for key in keys)
comb = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

returns_, Avg_Return_Short, Avg_Return_Long, Avg_Sum = [], [], [], []

for z in range(len(comb)):
    # apply trading strategy
    k_entry = comb[z]["k_entry"] # percentage of peaks and througs
    k_exit = comb[z]["k_exit"]  # solo funciona entre 0.23 y 0.53 cuando vale 0.75, esto es porque no hay signal en ese año
    EMA_days_12 = comb[z]["EMA_days_12"]
    EMA_days_26 = comb[z]["EMA_days_26"]
    STD_rw = comb[z]["STD_rollingwindow"]
    MXMN_rw = comb[z]["MAXMIN_rollingwindow"]
    strategy = histogram_retracement(stock, dataframe, k_entry, k_exit, EMA_days_12, EMA_days_26, STD_rw, MXMN_rw)
    strategy.signal_construction()
    ret, Short, Long, Sum = strategy.get_returns()
    returns_.append(ret)
    Avg_Return_Short.append(Short)
    Avg_Return_Long.append(Long)
    Avg_Sum.append(Sum)
    print(f"Iteración {z} de {len(comb) - 1}", end="\r")


print(f"""El maximo retorno medio de short es: {max(Avg_Return_Short)} 
se da para el indice: {Avg_Return_Short.index(max(Avg_Return_Short))}
y para la combinacion: {comb[Avg_Return_Short.index(max(Avg_Return_Short))]}
La tabla de retornos es: """)# returns_
returns_[Avg_Return_Short.index(max(Avg_Return_Short))]

print(f"""El maximo retorno medio de long es: {max(Avg_Return_Long)} 
se da para el indice: {Avg_Return_Long.index(max(Avg_Return_Long))}
y para la combinacion: {comb[Avg_Return_Long.index(max(Avg_Return_Long))]}
La tabla de retornos es: """)# returns_
print("\n")
returns_[Avg_Return_Long.index(max(Avg_Return_Long))]

print(f"""El maximo retorno medio conjunto es: {max(Avg_Sum)} 
se da para el indice: {Avg_Sum.index(max(Avg_Sum))}
y para la combinacion: {comb[Avg_Sum.index(max(Avg_Sum))]}
La tabla de retornos es: """)# returns_
print("\n")
returns_[Avg_Sum.index(max(Avg_Sum))]