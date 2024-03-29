import keras
from keras.models import load_model

from agent import Agent
from help_function import *
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def evaluate(stock_name, model_name):
    print("Start Evaluate")
    model = load_model("models/" + model_name)
    window_size = model.layers[0].input.shape.as_list()[1]

    agent = Agent(window_size, True, model_name)
    stock_data = getStockData(stock_name, "2018-03-01", "2019-06-01")
    data = stock_data['Close']
    length = len(data) - 1
    batch_size = 32

    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []

    fig, ax = plt.subplots()
    timeseries_iter = 0
    plt_buy = []
    plt_sell = []
    plt_original = []

    print("Start")
    for t in range(length):
        action = agent.act(state)
        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        plt_original.append((timeseries_iter, data[t]))
        # buy
        if action == 1:
            agent.inventory.append(data[t])
            plt_buy.append((timeseries_iter, data[t]))
            #print("Buy: " + str(data[t]))
        # sell
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            plt_sell.append((timeseries_iter, data[t]))
            #print("Sell: " + str(data[t]) + " | Profit: " + str(data[t] - bought_price))
        
        timeseries_iter += 1
        done = True if t == length - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("---------------------------")
            print(stock_name + " Total Profit: " + str(total_profit))
            print("---------------------------")
        
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    plt_original = np.array(plt_original)
    plt_buy = np.array(plt_buy)
    plt_sell = np.array(plt_sell)
    ax.plot(plt_original[:, 0], plt_original[:, 1], 'k', label='Original')
    ax.plot(plt_buy[:, 0], plt_buy[:, 1], 'ro', label="Buy")
    ax.plot(plt_sell[:, 0], plt_sell[:, 1], 'go', label="Sell")
    legend = ax.legend(loc='upper right')
    plt.show()

def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py [stock] [model]")
        exit()
    try:
        evaluate(sys.argv[1], sys.argv[2])
    except Exception as e:
        print("Error is: " + e)
    finally:
        exit()

if __name__ == "__main__":
    main()
