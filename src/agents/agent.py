import random

class Agent:

    def __init__(self, state_size, action_size, seed=42):
        """
        Initialize the Random agent with state and action sizes.

        Parameters:
        - state_size (int): The size of the state space.
        - action_size (int): The size of the action space.
        - seed (int): random seed for reproducibility
        """
        self.state_size = state_size
        self.action_size = action_size
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
        