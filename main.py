import argparse
import gym
import matplotlib.pyplot as plt

# FrozenLake-v0, MountainCar-v0, FreewayRAM-v0
ENV = None

def load_environment(name):
    global ENV
    ENV = gym.make(name)


def select_action(state):
    return ENV.action_space.sample()


def run_iterations(args):
    state = ENV.reset()
    reward_per_iteration = []
    for i in range(args.max_iterations):
        print("Step {}".format(i))
        reward_per_episode = []
        for step in range(args.max_steps):
            if args.render: ENV.render()
            action = select_action(state)
            obs, reward, done, _ = ENV.step(action) # take a random action
            reward_per_episode.append(reward)
            if done: break
        reward_per_iteration.append(reward_per_episode)
    return reward_per_iteration


def plot_rewards(rewards):
    plt.plot(range(len(rewards)), [sum(reward) for reward in rewards])
    plt.xlabel("#Iterations")
    plt.ylabel("Reward")
    plt.show()


def main(args):
    load_environment(args.env)
    reward_per_iteration = run_iterations(args)
    ENV.close()

    plot_rewards(reward_per_iteration)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='FrozenLake-v0', help="FrozenLake-v0, MountainCar-v0 or Freeway-RAM-v0")
    parser.add_argument('--render', type=bool, default=False, help="Render environment or not")
    parser.add_argument('--max-iterations', type=int, default=1000, help="Maximum amount of iterations")
    parser.add_argument('--max-steps', type=int, default=1000, help="Maximum amount of steos in an iteration")
    args = parser.parse_args()

    main(args)
