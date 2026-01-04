#
#   Script for plotting results from training history. 
#

import numpy as np
import matplotlib.pyplot as plt
import argparse

def draw_reward_graphs(data, y_axis, x_axis, labels, title, smoothing):
    fig, ax = plt.subplots(figsize=(6, 4))

    for i in range(0, len(data)):
        y_axis_data = [entry[y_axis] for entry in data[i]]
        x_axis_data = [entry[x_axis] for entry in data[i]]

        y_axis_data = np.convolve(y_axis_data, np.ones(smoothing)/smoothing, mode='valid')
        x_axis_data = x_axis_data[len(x_axis_data) - len(y_axis_data):]

        ax.plot(x_axis_data, y_axis_data, linestyle='-', label=labels[i])

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f'{title}, smoothing = {smoothing}')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

parser = argparse.ArgumentParser(description="This program draws graphs from evaluation history.")
parser.add_argument('--smoothing', type=int, required=True, help="Size of smoothing window")

args = parser.parse_args()

data_base = np.load("models/flappy_dqn/eval_history.npz", allow_pickle=True)
data_ddqn = np.load("models/flappy_ddqn/eval_history.npz", allow_pickle=True)
data_dueling_ddqn = np.load("models/flappy_dueling_ddqn/eval_history.npz", allow_pickle=True)

evaluations = data_base["evaluations"]
rollouts = data_base["rollouts"]

draw_reward_graphs(
    data=[data_base["evaluations"], data_ddqn["evaluations"], data_dueling_ddqn["evaluations"]],
    y_axis='ep_mean_reward',
    x_axis='total_timesteps',
    labels=["DQN", "DDQN", "Dueling DDQN"],
    title="evaluation",
    smoothing=args.smoothing
   )

plt.savefig("eval_comparison.png", format='png', dpi=300, bbox_inches='tight')

draw_reward_graphs(
    data=[data_base["rollouts"], data_ddqn["rollouts"], data_dueling_ddqn["rollouts"]],
    y_axis='ep_mean_reward',
    x_axis='total_timesteps',
    labels=["DQN", "DDQN", "Dueling DDQN"],
    title="training",
    smoothing=args.smoothing
   )

plt.savefig("train_comparison.png", format='png', dpi=300, bbox_inches='tight')

plt.show()