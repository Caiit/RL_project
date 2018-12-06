import gym

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
