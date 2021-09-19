from AdvertisingEnvironment.BiddingEnvironment import *
from AdvertisingEnvironment.Learner import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GPTS_Learner(Learner):
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(n_arms)
        self.sigmas = np.ones(n_arms) * 10
        self.revenue_probs = np.ones(10)
        self.pulled_arms = []
        alpha = 10.0
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha = alpha**2, normalize_y=True, n_restarts_optimizer=10)

    
    def update_observations(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward)
        self.pulled_arms.append(self.arms[pulled_arm])
        
    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x,y)
        
        x_pred = np.atleast_2d(self.arms).T
        self.means, self.sigmas = self.gp.predict(x_pred, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)
        
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()
        
    
    def pull_arm(self):
        bool_arr = self.revenue_probs > 0.20
        idxs = np.where(bool_arr)[0]
        ms = [self.means[i] for i in idxs]
        sigs = [self.sigmas[i] for i in idxs]
        idx = np.argmax(np.random.normal(ms, sigs))
        return idxs[idx]