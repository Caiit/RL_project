import gym
import argparse
import numpy as np

def get_all_states(env_name):
    env_demo = {"FrozenLake-v0": [4, 8, 9, 13, 14],
        "MountainCar-v0":[0]*50 + [2]*30 + [0]*50 + [2]*40,
        "FreewayNoFrameskip-v0": [1]*43}

    if env_name == "FrozenLake-v0":
        return env_demo[env_name]
    elif env_name == "MountainCar-v0":
        states = np.load('states_mountaincar.npy').tolist()
        return [states[i] for i in range(0, len(states), 10)]
    elif env_name == "FreewayNoFrameskip-v0":
        states = np.load('states_freeway.npy').tolist()
        return [states[i] for i in range(0, len(states), 2)]

    return None


def get_start_state(env, env_name, start_n):
    # Let's define start_n as 10 different states we can start in
    # n=1 means we start in the beginning of a rollout
    # n=9 means we start in the end of a rollout
    env.reset()
    env_demo = {"FrozenLake-v0": [4, 8, 9, 13, 14],
        "MountainCar-v0":[0]*50 + [2]*30 + [0]*50 + [2]*50,
        "Freeway-ramDeterministic-v0": [1]*43}

    if env_name == "FrozenLake-v0":
        states = env_demo[env_name]
        return states[min(start_n, len(states) - 1)]
    elif env_name == "MountainCar-v0":
        states = np.load('states_mountaincar.npy')
        return states[min(start_n, len(states) - 10)]
    elif env_name == "FreewayNoFrameskip-v0":
        return None
        states = np.load('states_freeway.npy')
        # TODO: not working yet
        env.state = states[min(start_n, len(states) - 5)]
        return env.state
    return None


def main(args):

    env_name = args.env
    start = int(args.start)
    #nvs = ['FrozenLake-v0', 'MountainCar-v0', 'Freeway-ram-v0']
    env = gym.make(env_name)
    env.reset()

    if env_name == "FrozenLake-v0":
        states = [4, 8, 9, 13, 14]
        env.env.s = states[start]
    elif env_name == "MountainCar-v0":
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
    parser.add_argument("env", help="which environment")
    parser.add_argument("start", help="how far from goal start")
    args = parser.parse_args()

    main(args)
