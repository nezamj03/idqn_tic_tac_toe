from typing import List
import numpy as np
import torch
from torch import nn
from collections import deque
import random
import matplotlib.pyplot as plt

class Memory():
    """ Represents memory deque """
    def __init__(self, max_length = 50):
        self.memory = deque([], maxlen = max_length)
    
    def append(self, item):
        self.memory.append(item)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

def state_to_dqn_input(observation : np.array):
    flattened = observation.flatten()
    return torch.tensor(flattened, dtype=torch.float32)

def sync(steps, syncing, networks : List[nn.Module]):
    policy, target = networks
    if steps > syncing:
        target.load_state_dict(policy.state_dict())
        return 0
    return steps

def plot_dictionary(dictionary):
    for k, v in dictionary.items():
        plt.plot(v, label = k)
    plt.legend()
    
def draw_state(state, agent_id):
    icon = {'player_1' : 'x', 'player_2' : 'o'}
    opp = {'x' : 'o', 'o' : 'x'}
    x = state[:, :, 0]
    o = state[:, :, 1]
    board = [['_' for _ in range(3)] for _ in range(3)]
    for row in range(3):
        for col in range(3):
            if x[row][col] == 1: board[row][col] = icon[agent_id]
            if o[row][col] == 1: board[row][col] = opp[icon[agent_id]]
    for row in range(3):
        print(' '.join(board[row]))
