import numpy as np
import math

def fun(bid):
    meanNrClick = math.trunc(500*math.tanh(bid))
    nrClick= math.trunc(np.random.uniform(meanNrClick-0.05*meanNrClick, meanNrClick+0.05*meanNrClick,1)[0])
    return nrClick

class BiddingEnvironment():
    def __init__(self, bids, sigma): #array of possible bids and (sigma) standard deviation of the reward function and we assume.
        self.bids = bids
        self.means = [fun(bid) for bid in bids]#mean of reward function
        self.sigmas = np.ones(len(bids)) * sigma

    def round(self, pulled_arm):
        reward = np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm], 1)
        return reward