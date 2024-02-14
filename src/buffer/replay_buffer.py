import random
import numpy as np
import torch
from collections import namedtuple, deque

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed=42):
        """
        Initializes a ReplayBuffer.

        Parameters:
        - action_size (int): The size of the action space.
        - buffer_size (int): Maximum size of the buffer.
        - batch_size (int): Size of each training batch.
        - seed (int): Random seed for reproducibility.
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # Use deque for efficient FIFO
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', ['state', 'action', 'next_state', 'reward', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """
        Adds an experience to the memory.

        Parameters:
        - state (array-like): The observed state.
        - action (int): The action taken.
        - reward (float): The reward received.
        - next_state (array-like): The next state observed.
        - done (bool): Whether the episode is complete.
        """
        e = self.experience(state, action, next_state, reward, done)
        self.memory.append(e)

    def sample(self):
        """
        Samples a batch of experiences from the memory.

        Returns:
        A tuple containing tensors for states, actions, rewards, next_states, and dones.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Returns the current size of the internal memory.
        """
        return len(self.memory)
