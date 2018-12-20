from common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np


"""
Plot the training results of different experiments in one plot with the variance over multiple seeds.
"""
colours = ['seagreen', 'darkviolet', 'goldenrod', 'slategray', 'tomato',  'chocolate', 'slategray', 'paleturquoise']

# Plot different percentages
experiments = ["20", "50", "90"]
labels = experiments

# Plot different starting states
# experiments = ["50", "50_more_states"]
# labels = ["17 states", "34 states"]

plot_type = "original_rewards"

n_steps = 200000
for c, n in enumerate(experiments):
    print("#", n)
    results1 = pu.load_results('./new_logs/mountain_car/' + n)[0]
    results2 = pu.load_results('./new_logs/mountain_car_2/' + n)[0]
    results3 = pu.load_results('./new_logs/mountain_car_3/' + n)[0]
    results4 = pu.load_results('./new_logs/mountain_car_4/' + n)[0]
    results5 = pu.load_results('./new_logs/mountain_car_5/' + n)[0]


    goals = []
    starts = []
    x_range = [0] * n_steps
    for r in [results1, results2, results3, results4, results5]:
        start_changes = [int(s) for s in list(r.progress.start_changes)[-1].split("_")]

        if plot_type == "original_rewards":
            if len(r.monitor.l) < len(x_range): x_range = r.monitor.l
            goals.append(pu.smooth(r.monitor.r, mode='causal', radius=200))
        else:
            reached_goal = [int(g) for g in list(r.progress.reached_goal)[-1].split("_")]
            rewards = np.zeros(n_steps)
            rewards[reached_goal] = 1
            goals.append(pu.smooth(rewards, mode='causal', radius=1000))
        starts.append(start_changes)

    mean = []
    std = []
    for i in range(len(x_range)):
        current_rewards = [goals[0][i], goals[1][i], goals[2][i], goals[3][i], goals[4][i]]
        mean.append(np.mean(current_rewards))
        std.append(np.std(current_rewards))
    mean_start = []
    for j in range(max([len(s) for s in starts])):
        current_starts = []
        for start in starts:
            if j < len(start):
                current_starts.append(start[j])
        mean_start.append(np.mean(current_starts))

    plt.plot([mean_start, mean_start], [[0] * len(mean_start), [-200] * len(mean_start)], alpha=0.4, color=colours[c])
    plt.plot(np.cumsum(x_range), mean, color=colours[c], label=labels[c])
    plt.fill_between(np.cumsum(x_range), (np.array(mean) - np.array(std)), (np.array(mean) + np.array(std)), alpha=0.2, color=colours[c])

plt.xlabel("#Iterations")
plt.ylabel("Returns")
plt.legend()
plt.show()

