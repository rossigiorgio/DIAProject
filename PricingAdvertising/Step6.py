import numpy as np
import matplotlib.pyplot as plt
from AdvertisingEnvironment.GPTS_Learner import *
from AdvertisingEnvironment.BiddingEnvironment import *
from ProjectEnviroment.projectEnviroment import *
from pricingEnviroment.Enviroment import *
from pricingEnviroment.TS_Learner import *

context = contextEnv()
prices = context.prices
probs = context.probabilities
max_price = np.max(prices)
cum_probs = np.zeros(10)
for i in range(0,9):
    cum_probs[i] = (probs[0][i] + probs[1][i] + probs[2][i]) / 3

n_arms = 10
T_days = 365

ts_rewards = []
totalRevenueTS = []
dailyTS = 0
cum_rewardTS = 0 


min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_arms)
sigma = 10
profit_margin = 0.15
rp = np.zeros(n_arms)
rn = np.zeros(n_arms)


p_env = Enviroment(n_arms = n_arms, probabilities = cum_probs)
b_env = BiddingEnvironment(bids=bids, sigma=sigma)
ts_learner = TS_Learner(n_arms = n_arms)
gpts_learner = GPTS_Learner(n_arms=n_arms, arms=bids)


for t in range(0, T_days):
    pulled_bid_arm = gpts_learner.pull_arm()
    bid_reward = b_env.round(pulled_bid_arm)
    
    pulled_price_arm = ts_learner.pull_arm()
    no_of_clicks = bid_reward * cum_probs[pulled_price_arm]
    cost_per_click = context.costPerClick(bids[pulled_bid_arm])

    for k in range(0, int(bid_reward)):
        price_reward = p_env.round(pulled_price_arm)
        cum_rewardTS += price_reward * (prices[pulled_price_arm] / max_price)
        dailyTS += price_reward * prices[pulled_price_arm] - cost_per_click
    
    totalRevenueTS.append(dailyTS)
    ts_rewards.append(cum_rewardTS / int(bid_reward))
    ts_learner.update(pulled_price_arm , cum_rewardTS / int(bid_reward))
    dailyTS = 0
    cum_rewardTS = 0

    profit = prices[pulled_price_arm] * profit_margin * no_of_clicks
    if(profit > (cost_per_click * bid_reward)):
        rp[pulled_bid_arm] += 1
        gpts_learner.revenue_probs[pulled_bid_arm] = (rp[pulled_bid_arm]) /  (rp[pulled_bid_arm] + rn[pulled_bid_arm])
    else: 
        rn[pulled_bid_arm] += 1
        gpts_learner.revenue_probs[pulled_bid_arm] = (rp[pulled_bid_arm]) /  (rp[pulled_bid_arm] + rn[pulled_bid_arm])
    gpts_learner.update(pulled_bid_arm, no_of_clicks)



plt.figure(0)
plt.xlabel("t")
plt.ylabel("Revenue")
plt.plot(np.cumsum(totalRevenueTS, axis=0), 'r')
plt.legend(["TS"])
plt.show()