import torch
import torch.nn as nn
import gym
import random

class PolicyNetwork(nn.Module):
    
    def __init__(self, state_size=4, action_size=2, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(state_size, num_hidden)
        self.l2 = nn.Linear(num_hidden, action_size)

    def forward(self, x):
        h = self.l1.forward(x)
        a = torch.relu(h)
        return self.l2.forward(a)

def get_epsilon(it):
    if it > 1000: return 0.05
    return 1 - 0.00095*it

def select_action(model, state, epsilon):
    # Check if we need to take a random action or not
    if random.random() < epsilon:
        return random.choice([0, 1])
    with torch.no_grad():
        actions = model(torch.tensor(state, dtype=torch.float))
        _, index = torch.max(actions, 0)
    return index.item()

def main():
	env = gym.envs.make("CartPole-v0")
	model = PolicyNetwork(4,2)
	s = env.reset() 
	a = select_action(model, s, 0.05)
if __name__ == "__main__":
    main()    