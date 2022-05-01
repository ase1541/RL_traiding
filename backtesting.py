from main import params, dataframe, stock
from histogram_retracement import histogram_retracement
import itertools
# Define all posible combinations for the backtesting
keys = params.keys()
values = (params[key] for key in keys)
comb = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

returns_, Avg_Sharp_Short, Avg_Sharp_Long, Avg_Sharp_sum = [], [], [], []

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
    Avg_Sharp_Short.append(Short)
    Avg_Sharp_Long.append(Long)
    Avg_Sharp_sum.append(Sum)
    print(f"Iteración {z} de {len(comb) - 1}", end="\r")


print(f"""El maximo sharp medio de short es: {max(Avg_Sharp_Short)} 
se da para el indice: {Avg_Sharp_Short.index(max(Avg_Sharp_Short))}
y para la combinacion: {comb[Avg_Sharp_Short.index(max(Avg_Sharp_Short))]}
La tabla de retornos es: """)# returns_
returns_[Avg_Sharp_Short.index(max(Avg_Sharp_Short))]

print(f"""El maximo sharp medio de long es: {max(Avg_Sharp_Long)} 
se da para el indice: {Avg_Sharp_Long.index(max(Avg_Sharp_Long))}
y para la combinacion: {comb[Avg_Sharp_Long.index(max(Avg_Sharp_Long))]}
La tabla de retornos es: """)# returns_
print("\n")
returns_[Avg_Sharp_Long.index(max(Avg_Sharp_Long))]

print(f"""El maximo sharp medio conjunto es: {max(Avg_Sharp_sum)} 
se da para el indice: {Avg_Sharp_sum.index(max(Avg_Sharp_sum))}
y para la combinacion: {comb[Avg_Sharp_sum.index(max(Avg_Sharp_sum))]}
La tabla de retornos es: """)# returns_
print("\n")
returns_[Avg_Sharp_sum.index(max(Avg_Sharp_sum))]