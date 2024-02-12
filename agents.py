from network import DQN
import numpy as np
import random 
import torch
from utils import Memory, state_to_dqn_input

class Agent:
    """ Abstract Agent"""
    def __init__(self):
        self.memory = Memory()

    def action(self, action_mask : np.array, **kwargs):
        raise NotImplementedError
    
    
class Random(Agent):
    """ Random Agent """

    def __init__(self):
        super().__init__()
    
    def action(self, action_mask : np.array, **kwargs):
        possible_actions = np.arange(len(action_mask))[action_mask == 1]
        return random.choice(possible_actions)
    
class DQNAgent(Agent):
    """ DQN Agent """

    def __init__(self, policy_network, target_network):
        super().__init__()
        self.policy_network : DQN = policy_network # policy and target network
        self.target_network : DQN = target_network

    def action(self, action_mask : np.array, **kwargs):
        if 'input' not in kwargs or 'epsilon' not in kwargs:
            raise TypeError('input and epsilon are required kwargs')
        input, epsilon = kwargs['input'], kwargs['epsilon']
        # epsilon-greedy 
        if random.random() < epsilon: # exploration
            possible_actions = np.arange(len(action_mask))[action_mask == 1]
            return random.choice(possible_actions)
        with torch.no_grad(): #exploitation
            q_values = self.policy_network(state_to_dqn_input(input))
            return q_values.argmax().item()