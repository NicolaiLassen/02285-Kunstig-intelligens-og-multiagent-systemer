import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_i():
    rewards = torch.load('ckpt/intrinsic_rewards.ckpt')

    x = np.array([i * 2000 * 4 for i in range(len(rewards))])
    y = rewards.numpy()

    # Don't smooth ends
    sy = smooth(y, 12)
    sy[:4] = rewards[:4]
    sy[len(sy) - 4:len(sy)] = rewards[len(sy) - 4:len(sy)]

    # Don't smooth ends
    sy = smooth(y, 12)
    sy[:4] = rewards[:4]
    sy[len(sy) - 4:len(sy)] = rewards[len(sy) - 4:len(sy)]

    d_smooth = {'Steps': x, 'log2 Rewards': sy}
    df_smooth = pd.DataFrame(data=d_smooth)

    d = {'Steps': x, 'log2 Rewards': y}
    df = pd.DataFrame(data=d)

    # #FF0000
    # #ffcc66
    sns.lineplot(x="Steps", y="log2 Rewards", data=df, color='#ffcc66', alpha=0.3, linewidth='2.6')
    sns.lineplot(x="Steps", y="log2 Rewards", data=df_smooth, color='#ffcc66', linewidth='1.5')
    plt.title("PPO training intrinsic rewards")
    plt.yscale('log', base=2)
    plt.show()


if __name__ == '__main__':
    plot_i()
    with open('ckpt/reward_level.ckpt', 'rb') as handle:
        b = pickle.load(handle)
    print(b)
