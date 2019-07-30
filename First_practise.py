import pandas_datareader as pdr
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import random
import pandas as pd
#import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import datetime
import lstm
import time
import csv
import os
import DecisionPolicy

def print_head(stock): #print the first 5 rowsm 
    print(stock.head())

def print_tail(stock): #print the last 5 rows
    print(stock.tail())

def print_summary(stock): #print the summary for the whole datum
    print(stock.describe())

def print_index(stock): #print the timestamp for each data
    print(stock.index)

def print_column(stock): #Print the Columns Header
    print(stock.columns)

def print_subColumn(stock, index, count): #Print the item from [(count >0)? Top : End] with the Index
    print(stock[index][count:])

def print_label_based_index(stock, index): #Print the item with label based index. E.g. index = '2007' All items with date from year 2007
    print(stock.loc[index])

def print_latter_index(stock, start, end): #Print from the Start to End items
    print(stock.iloc[start:end])

def split_train_test(dataset, ratio):
    train, test = [], []
    length = round(len(dataset) * ratio + 0.5)
    train, test = dataset[0: length], dataset[length:]
    return train, test

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

def main():
    outdir = './data'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    aapl = pdr.get_data_yahoo('AAPL', start=datetime.datetime(2006, 10, 1), end=datetime.datetime(2019, 1, 1))
    aapl.to_csv('data/aapl_ohlc.csv')
    df = pd.read_csv('data/aapl_ohlc.csv', index_col='Date', header=0, parse_dates=True)

    close = df['Close']
    actions = ['Buy', 'Sell', 'Hold']
    hist = 200
    #policy = RandomDecisionPolicy(actions)
    #policy = QLearningDecisionPolicy(actions)
    budget = 1000.0
    num_stocks = 0
    avg, std = run_simulation(policy, budget, num_stocks, price, hist)
    print(avg, std)
    
if __name__=="__main__":
    main()