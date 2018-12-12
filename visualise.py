from common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np
import itertools

results = pu.load_results('./logs/mountain_car')

colours = ['seagreen', 'paleturquoise', 'tomato', 'darkviolet', 'chocolate', 'slategray', 'goldenrod']
for i, r in enumerate(results):
    reached_goal = [int(g) for g in list(r.progress.reached_goal)[-1].split("_")]
    rewards = np.zeros(max(np.cumsum(r.monitor.l)))
    rewards[reached_goal] = 1
    plt.plot(range(max(np.cumsum(r.monitor.l))), pu.smooth(rewards, mode='causal', radius=200), c=colours[i])
    start_changes = [int(s) for s in list(r.progress.start_changes)[-1].split("_")]
    plt.plot([start_changes, start_changes],
             [[0] * len(start_changes), [1] * len(start_changes)],
             color='b', alpha=0.3)
plt.xlabel("#Iterations")
plt.ylabel("Rewards")
plt.legend()
# plt.show()
plt.savefig("./results/plot.png")
