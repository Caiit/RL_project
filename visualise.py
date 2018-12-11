from common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np
import itertools

results = pu.load_results('./logs/mountain_car_ppo2')

colours = ['seagreen', 'paleturquoise', 'tomato', 'darkviolet', 'chocolate', 'slategray', 'goldenrod']
for i, r in enumerate(results):
    plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10), c=colours[i], label=str(i))
    start_changes = list(r.progress.start_change)
    plt.plot([start_changes, start_changes], [[min(r.monitor.r)] * len(start_changes),
                                              [max(r.monitor.r)] * len(start_changes)],
             color='b', alpha=0.3)
plt.xlabel("#Iterations")
plt.ylabel("Rewards")
plt.legend()
plt.show()
