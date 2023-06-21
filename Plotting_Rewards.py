import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean
import numpy as np


def plot(raw_rewardsperep, save_path):
    plt.grid(linestyle='-.')
    rewards_smoothed = pd.Series(raw_rewardsperep).rolling(100, min_periods=10).mean()
    rewards_mean = np.mean(raw_rewardsperep)*np.ones(len(raw_rewardsperep))

    #plt.plot(raw_rewardsperep, linewidth=0.5, label='reward per episode')
    plt.plot(rewards_smoothed, linewidth=2.0, label='smoothed reward (over window size=10)')
    plt.plot(rewards_mean, linewidth=2.0)
    plt.legend(["Smoothed Reward (Window size = 100)", "Average Reward across all Episodes = " + str(np.mean(raw_rewardsperep))])
    plt.xlabel("Training Episode")
    plt.ylabel("Total Episode Reward")
    plt.title("Total Episode Reward v.s. Training Episode")
    plt.savefig(save_path)