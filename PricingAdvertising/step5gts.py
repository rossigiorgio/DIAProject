import numpy as np
import matplotlib.pyplot as plt
from AdvertisingEnvironment.GTS_Learner import *
from AdvertisingEnvironment.BiddingEnvironment import *
from ProjectEnviroment.projectEnviroment import *

n_arms = 10
min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_arms)
sigma = 10

context = contextEnv()
fixed_price = np.mean(context.prices)
profit_margin = fixed_price * 0.15
conv_prob = np.mean(sum(context.probabilities, []))
rp = np.zeros(10)
rn = np.zeros(10)

T = 365
gts_rewards_per_experiment = []
env = BiddingEnvironment(bids=bids, sigma=sigma)
gts_learner = GTS_Learner(n_arms=n_arms)
u_idx = 0
for i in range(0,10):
    for j in range(0,4):
        reward = env.round(u_idx)
        u_idx += 1
        u_idx = u_idx % 10
        pulled_arm = u_idx
        reward = env.round(pulled_arm)
        nof = reward * conv_prob
        cost = context.costPerClick(env.bids[pulled_arm]) * reward
        profit = profit_margin * nof
        if(profit > cost):
            rp[pulled_arm] += 1
            gts_learner.revenue_probs[pulled_arm] = (rn[pulled_arm]) /  (rp[pulled_arm] + rn[pulled_arm])
        else: 
            rn[pulled_arm] += 1
            gts_learner.revenue_probs[pulled_arm] = (rn[pulled_arm]) /  (rp[pulled_arm] + rn[pulled_arm])

        gts_learner.update(pulled_arm, reward) 


for t in range(10, T):   
    for j in range(0,4):
        #Gaussian Process Thompson Sampling Learner
        pulled_arm = gts_learner.pull_arm()
        reward = env.round(pulled_arm)
        nof = reward * conv_prob
        cost = context.costPerClick(env.bids[pulled_arm]) * reward
        profit = profit_margin * nof 
        if(profit > cost):
            rp[pulled_arm] += 1
            gts_learner.revenue_probs[pulled_arm] = (rn[pulled_arm]) /  (rp[pulled_arm] + rn[pulled_arm])
        else: 
            rn[pulled_arm] += 1
            gts_learner.revenue_probs[pulled_arm] = (rn[pulled_arm]) /  (rp[pulled_arm] + rn[pulled_arm])
        
        gts_learner.update(pulled_arm, reward)

gts_rewards_per_experiment.append(gts_learner.collected_rewards)

opt = np.max(env.means)
plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt - gts_rewards_per_experiment, axis=0)), 'r')
plt.legend(["GTS"])
plt.show()