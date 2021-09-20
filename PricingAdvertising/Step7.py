import numpy as np
import matplotlib.pyplot as plt
from AdvertisingEnvironment.GTS_Learner import *
from AdvertisingEnvironment.BiddingEnvironment import *
from ProjectEnviroment.projectEnviroment import *
from pricingEnviroment.Enviroment import *
from pricingEnviroment.TS_Learner import *

context= contextEnv()
prices = context.prices
max_price = np.max(prices)



probs = context.probabilities
# fixedBid = 1.0

optClass1 = np.max(probs[0])
optClass2 = np.max(probs[1])
optClass3 = np.max(probs[2])

n_arms = 10;
T = 365

ts_rewardsC1 = [] 
ts_rewardsC2 = [] 
ts_rewardsC3 = [] 

totalRevenueTS = []
totalRevenueSingle = []

dailyTS = 0
dailyTSSingle = 0


cumRewardTSC1 = 0
cumRewardTSC2 = 0 
cumRewardTSC3 = 0
cumRewardSingle = 0

min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_arms)
sigma = 10
profit_margin = 0.15
rp = np.zeros(n_arms)
rn = np.zeros(n_arms)

# nrClickPerClass=167


#for each class define a new enviroment with his respective probabilities
p_envClass1 = Enviroment(n_arms = n_arms, probabilities = probs[0])
p_envClass2 = Enviroment(n_arms = n_arms, probabilities = probs[1])
p_envClass3 = Enviroment(n_arms = n_arms, probabilities = probs[2])

# b_envClass1 = BiddingEnvironment(bids = bids, sigma = sigma)
# b_envClass2 = BiddingEnvironment(bids = bids, sigma = sigma)
# b_envClass3 = BiddingEnvironment(bids = bids, sigma = sigma)

b_env = BiddingEnvironment(bids = bids, sigma = sigma)

#for every class define 3 different TS learner
ts_learnerClass1 = TS_Learner(n_arms = n_arms)
ts_learnerClass2 = TS_Learner(n_arms = n_arms)
ts_learnerClass3 = TS_Learner(n_arms = n_arms)

ts_learnerSingle = TS_Learner(n_arms = n_arms)

# gts_learner1 = GTS_Learner(n_arms=n_arms, arms=bids)
# gts_learner2 = GTS_Learner(n_arms=n_arms, arms=bids)
# gts_learner3 = GTS_Learner(n_arms=n_arms, arms=bids)

gts_learner = GTS_Learner(n_arms=n_arms)



for t in range (0,T):
    
    
    #pull TS arms
    p_pulled_armC1 = ts_learnerClass1.pull_arm() 
    p_pulled_armC2 = ts_learnerClass2.pull_arm() 
    p_pulled_armC3 = ts_learnerClass3.pull_arm() 
    p_pulled_armSingle = ts_learnerSingle.pull_arm()

    # b_pulled_armC1 = gts_learner1.pull_arm()
    # b_pulled_armC2 = gts_learner2.pull_arm()
    # b_pulled_armC3 = gts_learner3.pull_arm()
    # b_pulled_armSingle = gts_learnerSingle.pull_arm()
    b_pulled_arm = gts_learner.pull_arm()

    # bid_reward1 = b_envClass1.round(b_pulled_armC1)
    # bid_reward2 = b_envClass2.round(b_pulled_armC2)
    # bid_reward3 = b_envClass3.round(b_pulled_armC3)
    
    bid_reward = b_env.round(b_pulled_arm)


    no_of_clicks1 = int(bid_reward * probs[0][p_pulled_armC1]) + 1
    no_of_clicks2 = int(bid_reward * probs[1][p_pulled_armC2]) + 1
    no_of_clicks3 = int(bid_reward * probs[2][p_pulled_armC3]) + 1
    

    cost_per_click = context.costPerClick(bids[b_pulled_arm])
    
    #I pull a new arm every day, and i can distinguish the three different
    #class, and propose to each of them a different class
    
    #Class 1
    
    for x in range(0, no_of_clicks1):
        #Getting reward from his best arm
        reward = p_envClass1.round(p_pulled_armC1) #calcolo il reward
        cumRewardTSC1 += reward * (prices[p_pulled_armC1] / max_price)
        dailyTS += reward * prices[p_pulled_armC1] - cost_per_click
    
        #using single learner
        reward = p_envClass1.round(p_pulled_armSingle)    
        dailyTSSingle += reward * prices[p_pulled_armSingle] - cost_per_click
        cumRewardSingle += reward * (prices[p_pulled_armSingle] / max_price)
    
    
        
    #Class 2
    for x in range(0, no_of_clicks2):
        #Getting reward from his best arm
        reward = p_envClass2.round(p_pulled_armC2) #calcolo il reward
        cumRewardTSC2 += reward *(prices[p_pulled_armC2] / max_price)
        dailyTS += reward * prices[p_pulled_armC2] - cost_per_click
               
               
        #using single learner
        
        reward = p_envClass2.round(p_pulled_armSingle)
        dailyTSSingle += reward * prices[p_pulled_armSingle] - cost_per_click
        cumRewardSingle += reward * (prices[p_pulled_armSingle] / max_price)
        
     
    #Class 3
    for x in range(0, no_of_clicks3):
        #Getting reward from his best arm
        reward = p_envClass3.round(p_pulled_armC3) #calcolo il reward
        cumRewardTSC3 += reward * (prices[p_pulled_armC3] / max_price)
        dailyTS += reward * prices[p_pulled_armC3] - cost_per_click
                
        
        #using single learner
        
        reward = p_envClass3.round(p_pulled_armSingle)
        dailyTSSingle += reward * prices[p_pulled_armSingle] - cost_per_click
        cumRewardSingle += reward * (prices[p_pulled_armSingle] / max_price)
    
    
    #make the average of the cumulative reward 
    totalRevenueTS.append(dailyTS)
    totalRevenueSingle.append(dailyTSSingle)
    
    
    ts_rewardsC1.append(cumRewardTSC1/no_of_clicks1)  
    ts_rewardsC2.append(cumRewardTSC2/no_of_clicks2)  
    ts_rewardsC3.append(cumRewardTSC3/no_of_clicks3)  

    total_clicks = no_of_clicks1 + no_of_clicks2 + no_of_clicks3    
    ts_learnerClass1.update(p_pulled_armC1,cumRewardTSC1/no_of_clicks1)
    ts_learnerClass2.update(p_pulled_armC2,cumRewardTSC2/no_of_clicks2)
    ts_learnerClass3.update(p_pulled_armC3,cumRewardTSC3/no_of_clicks3)
    ts_learnerSingle.update(p_pulled_armSingle, cumRewardSingle/total_clicks)
    
    profit1 = prices[p_pulled_armC1] * profit_margin * no_of_clicks1
    profit2 = prices[p_pulled_armC2] * profit_margin * no_of_clicks2
    profit3 = prices[p_pulled_armC3] * profit_margin * no_of_clicks3
    total_profit = profit1 + profit2 + profit3
    if(total_profit > (cost_per_click * bid_reward * 3)):
        rp[b_pulled_arm] += 1
        gts_learner.revenue_probs[b_pulled_arm] = (rp[b_pulled_arm]) /  (rp[b_pulled_arm] + rn[b_pulled_arm])
    else: 
        rn[b_pulled_arm] += 1
        gts_learner.revenue_probs[b_pulled_arm] = (rp[b_pulled_arm]) /  (rp[b_pulled_arm] + rn[b_pulled_arm])
    gts_learner.update(b_pulled_arm, total_clicks)
    
    cumRewardTSC1=0
    cumRewardTSC2=0
    cumRewardTSC3=0
    cumRewardSingle=0
    

    dailyTS=0
    dailyTSSingle=0

print(np.cumsum(totalRevenueTS, axis=0)[364])
print(np.cumsum(totalRevenueSingle, axis=0)[364])




plt.figure(0)
plt.xlabel("t")
plt.ylabel("Revenue")
plt.plot(np.cumsum(totalRevenueTS, axis=0), 'r')
plt.plot(np.cumsum(totalRevenueSingle, axis=0), 'y')
plt.legend(["All C1", "Single"])



plt.show()
