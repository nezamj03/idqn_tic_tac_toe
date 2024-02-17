import random

class Agent:

    def __init__(self, seed=42):
        """
        Initialize the Random agent

        Parameters:
        - seed (int): random seed for reproducibility
        """
        self.seed = random.seed(seed)
    
    def act(self, state, action_mask, **kwargs):
        """
        Selects an action based on a given state and action mask.

        Parameters:
        - state (array-like): the current state of the environment (unused in this method).
        - action_mask (array-like): a mask that indicates valid actions with 1s.

        Returns:
        - int: a selected valid action.
        """
        raise NotImplementedError()

    def id(self, id):
        """
        Sets this agent's ID in a game

        Parameters:
        - id (str): agent id for games
        """
        self._id = id
        