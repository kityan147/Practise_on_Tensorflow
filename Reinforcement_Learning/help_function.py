import pandas_datareader as pdr
import pandas as pd
import numpy as np
import datetime
import math
import os

def getStockData(stock_name, start_date, end_date):
    outdir = './data'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if stock_name == "HSI":
        stock_name = "^HSI"
    stock = pdr.get_data_yahoo(stock_name, start=start_date, end=end_date)
    stock_name = 'data/' + stock_name + '.csv'
    stock.to_csv(stock_name)
    df = pd.read_csv(stock_name, index_col='Date', header=0, parse_dates=True)

    return df

# sigmoid function
def sigmoid(x):
    try:
        if x < 0:
            return 1 / (1 + math.exp(x))    
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return float(1 / (1 + math.exp(-x)))

# return an N-day state representation ending at time t
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else np.concatenate((-d * [data[0]], data[0:(t + 1)])) #pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return np.array([res])