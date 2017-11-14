import numpy as np

class OUNoise(object):

    def __init__(self, action_dimension,mu=0., theta=0.15, sigma=0.3, seed = None):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def seed(self, seed = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)