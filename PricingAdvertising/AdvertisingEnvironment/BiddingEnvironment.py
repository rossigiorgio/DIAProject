import numpy as np

def fun(x):
    return 100 * (1.0 - np.exp(-4*x+3*x**3)) #the function maps the bid corresponding the number of clicks

class BiddingEnvironment():
    def __init__(self, bids, sigma): #array of possible bids and (sigma) standard deviation of the reward function and we assume.
        self.bids = bids
        self.means = fun(bids) #mean of reward function
        self.sigmas = np.ones(len(bids)) * sigma

    def round(self, pulled_arm):
        reward = np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm], 1)
        return reward