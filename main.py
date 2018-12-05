import argparse
import gym
import matplotlib.pyplot as plt
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from policy import PolicyNetwork

# FrozenLake-v0, MountainCar-v0, FreewayRAM-v0
ENV = None

def load_environment(name):
    global ENV
    ENV = gym.make(name)


def to_tensor(state, state_size):
    '''
    Make one hot from FrozenLake, the rest is already a vector.
    '''
    if args.env == "FrozenLake-v0":
        one_hot = torch.zeros(state_size)
        one_hot[state] = 1
        return one_hot
    return torch.tensor(state, dtype=torch.float)


def get_epsilon(it):
    if it > 1000: return 0.05
    return 1 - 0.00095*it


def select_action(model, state, epsilon, n_actions):
    # Check if we need to take a random action or not
    if random.random() < epsilon:
        return random.choice(range(n_actions))
    with torch.no_grad():
        q_values = model(state)
        index = torch.max(q_values, 0)[1]
        return index.item()


def compute_q_val(model, state, action):
    q_values = model(state)
    return q_values[action]


def compute_target(model, reward, next_state, done, discount_factor):
    if done:
        # TODO: fix?
        max_q_values = torch.zeros(1)
    else:
        q_values = model(next_state)
        max_q_values = torch.max(q_values, 0)[0]
    return reward + discount_factor * max_q_values


def run_iterations(args):
    # Init model
    state_size = 16
    action_size = 4
    if args.env == "MountainCar-v0":
        state_size = 2
        action_size = 3
    if args.env == "Freeway-ram-v0":
        state_size = 128
        action_size = 3
    if args.env == "CartPole-v0":
        state_size = 4
        action_size = 2
    model = PolicyNetwork(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), args.learning_rate)

    reward_per_iteration = []
    for i in range(args.max_iterations):
        state = to_tensor(ENV.reset(), state_size)
        reward_per_episode = []
        episode_loss = 0
        for step in range(args.max_steps):
            if args.render: ENV.render()
            action = select_action(model, state, get_epsilon(i), action_size)
            next_state, reward, done, _ = ENV.step(action) # take a random action
            # compute the q value
            q_val = compute_q_val(model, state, action)


            with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
                next_state = to_tensor(next_state, state_size)
                target = compute_target(model, reward, next_state, done, args.discount_factor)

            # loss is measured from error between current and newly expected Q values
            loss = F.smooth_l1_loss(q_val, target)

            # backpropagation of loss to Neural Network (PyTorch magic)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_loss += loss
            state = next_state
            reward_per_episode.append(reward)
            if done: break

        if i % args.print_every == 0:
            print("Reward", reward, sum(reward_per_episode))
            print("Step {:6d} with loss: {:4f}".format(i, episode_loss))
        reward_per_iteration.append(reward_per_episode)
    return reward_per_iteration


def plot_rewards(rewards):
    plt.plot([sum(reward) for reward in rewards])
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
    parser.add_argument('--epsilon', type=float, default=0.01, help="Epsilon for epsilon greedy")
    parser.add_argument('--learning-rate', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--discount-factor', type=float, default=0.8, help="Discount factor")
    parser.add_argument('--print-every', type=int, default=10, help="Print status every x iteration")
    args = parser.parse_args()

    main(args)
