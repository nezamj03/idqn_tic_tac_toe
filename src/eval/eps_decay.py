import numpy as np

class LinearDecay:
    
    def __init__(self, episodes):
        self.episodes = episodes
    
    def get(self, eps, iter):
        return eps - 1/self.episodes

class ExponentialDecay:

    def __init__(self, decay):
        self.decay = decay
    
    def get(self, eps, iter):
        return np.exp(-self.decay * iter)

