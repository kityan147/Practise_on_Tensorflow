# LSTM
import pandas_datareader as pdr
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import random
import pandas as pd
#import tensorflow as tf
import numpy as np
import matplotlib
import datetime
import lstm
import time
import csv
import os

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
    window_size = 30
    predict_len = 15
    batch_size = 80
    epochs = 1
    x_train, y_train, x_test, y_test = lstm.load_data(close, window_size, True)

    #date_price = df['Close']
    #date_price = np.array(date_price).astype('float32')
    
    #train, test = split_train_test(date_price, 0.8)
    #x_train, y_train = create_dataset(train, 1)
    #x_test, y_test = create_dataset(test, 1)

    #x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    #x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    model = Sequential()

    model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print('compilation time:', time.time() - start)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.05)
    
    predictions = lstm.predict_sequences_multiple(model, x_test, window_size, predict_len)
    lstm.plot_results_multiple(predictions, y_test, predict_len)
if __name__=="__main__":
    main()