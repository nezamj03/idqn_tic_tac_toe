import random
import numpy as np

from .agent import Agent

class RandomAgent(Agent):
    def __init__(self, seed=42):
        super().__init__(seed=seed)
    
    def act(self, state, action_mask, **kwargs):
        """
        Selects a random action based on a given state and action mask.

        Parameters:
        - state (array-like): The current state of the environment (unused in this method).
        - action_mask (array-like): A mask that indicates valid actions with 1s.

        Returns:
        - int: A randomly selected valid action.
        """
        possible_actions = np.flatnonzero(action_mask == 1)
        return random.choice(possible_actions)