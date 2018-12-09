import gym
import argparse
import numpy as np

def get_all_states(env_name):
    env_demo = {"FrozenLake-v0": [4, 8, 9, 13, 14],
        "MountainCar-v0":[0]*50 + [2]*30 + [0]*50 + [2]*40,
        "Freeway-ramDeterministic-v0": [1]*43}
    return env_demo[env_name]


def get_start_state(env, env_name, start_n):
    env.reset()
    env_demo = {"FrozenLake-v0": [4, 8, 9, 13, 14],
        "MountainCar-v0":[0]*50 + [2]*30 + [0]*50 + [2]*40,
        "Freeway-ramDeterministic-v0": [1]*43}

    if env_name == "FrozenLake-v0":
        states = env_demo[env_name]
        #env.env.s = states[start_n]
        return states[start_n]
    elif env_name == "MountainCar-v0":
        states = np.load('states_mountaincar.npy')
        print(len(states))
        #env.env.state = states[start_n]
        return states[start_n]
    elif env_name == "Freeway-ramDeterministic-v0":
        states = np.load('states_freeway.npy')
        restored = states[start_n]
        return restored
    return None


def main(args):

    env_name = args.env
    start = args.start
    #nvs = ['FrozenLake-v0', 'MountainCarContinuous-v0', 'Freeway-ram-v0']
    env = gym.make(env_name)
    env.reset()

    if env_name == "FrozenLake-v0":
        states = [4, 8, 9, 13, 14]
        env.env.s = states[start]
    elif env_name == "MountainCarContinuous-v0":
        states = []
        action_list = [0]*50 + [2]*30 + [0]*50 + [2]*60
        for action in action_list:
            observation = env.step(action)
            done = observation[2]
            states.append(observation[0])
            if done:
                break
        env.env.state = states[start]
    elif env_name == "Freeway-ramDeterministic-v0":
        states = []
        action_list = [1]*43
        for action in action_list:
            observation =  env.step(action)
            done = observation[2]
            states.append(observation[0])
        env.env.state = states[start]

    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", help="which environment")
    parser.add_argument("start", help="how far from goal start")
    args = parser.parse_args()

    main(args)
