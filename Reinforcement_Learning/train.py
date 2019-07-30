from agent import Agent
from help_function import *
import sys

def train(stock_name, window_size, episode_count):
    agent = Agent(window_size)

    stock_data = getStockData(stock_name, '2015-01-01', '2019-03-01')
    
    data = stock_data['Close']

    length = len(data) - 1
    batch_size = 32

    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))
        state = getState(data, 0, window_size + 1)

        total_profit = 0
        agent.inventory = []

        for t in range(length):
            action = agent.act(state)

            # sit
            next_state = getState(data, t + 1, window_size + 1)
            reward = 0

            # buy
            if action == 1:
                agent.inventory.append(data[t])
                print("Buy:" + str(data[t]))
            # Sell
            elif action == 2 and len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price
                print("Sell: " + str(data[t]) + " | Profit: " + str(data[t] - bought_price))

            done = True if t == length - 1 else False
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("----------------------")
                print("Total Profit: " + str(total_profit))
                print("----------------------")

            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)

        if e % 10 == 0:
            agent.model.save("models/model_ep" + str(e))

def main():
    if len(sys.argv) != 4:
	    print("Usage: python train.py [stock] [window] [episodes]")
	    exit()

    train(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

if __name__ == "__main__":
    main()
