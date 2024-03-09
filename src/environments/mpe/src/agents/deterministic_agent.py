import numpy as np
import torch

from .....agents.agent import Agent

class DeterministicAgent(Agent):
    
    def __init__(self, state_size: int, action_size: int, seed: int = 42):
        super().__init__(seed)

    def act(self, state: torch.Tensor, action_mask: np.ndarray) -> int:
        
        relativex, relativey = state[:2]

        if np.sqrt(relativex**2 + relativey**2) < 0.5:
            return 0
        if np.abs(relativex) > np.abs(relativey):
            return 2 - int(relativex < 0)
        else:
            return 4 - int(relativey < 0)