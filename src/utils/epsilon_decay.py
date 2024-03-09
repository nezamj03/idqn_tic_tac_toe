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
    
class ConcaveDecay:

    def __init__(self, episodes, stretch=None, exploit=0.05):
        explore = int(episodes * (1 - exploit))
        if stretch is None: stretch = 10/explore
        self.epsilon = np.zeros(episodes)
        for i in range(explore):
            self.epsilon[i] = 2 / (1 + np.exp(stretch * (i - explore))) - 1
    
    def get(self, eps, iter):
        return self.epsilon[iter]

