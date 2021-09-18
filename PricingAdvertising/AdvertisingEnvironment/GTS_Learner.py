from AdvertisingEnvironment.BiddingEnvironment import *
from AdvertisingEnvironment.Learner import *

class GTS_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.means = np.zeros(n_arms)
        self.sigmas = np.ones(n_arms) * 1e3
        self.revenue_probs = np.zeros(n_arms)

        
    def pull_arm(self):
        bool_arr = self.revenue_probs > 0.20
        idxs = np.where(bool_arr)[0]
        ms = [self.means[i] for i in idxs]
        sigs = [self.sigmas[i] for i in idxs]
        idx = np.argmax(np.random.normal(ms, sigs))
        return idxs[idx]
    
     
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.means[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm])
        n_samples = len(self.rewards_per_arm[pulled_arm])
        if n_samples > 1:
            self.sigmas[pulled_arm] = np.std(self.rewards_per_arm[pulled_arm]) / n_samples
