# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:05:18 2021

@author: MorganaGiorgio
"""

from ProjectEnviroment.projectEnviroment import *
import numpy as np
from pricingEnviroment.Enviroment import *
from pricingEnviroment.TS_Learner import *
from pricingEnviroment.UCB1 import *
#from Learner import Learner
import matplotlib.pyplot as plt

context= contextEnv()
prezzi=context.prices

print(prezzi)

min_prezzo= np.min(prezzi)
max_prezzo= np.max(prezzi)

p=context.probabilities
c=np.zeros(10)
for i in range(0,9):
    c[i]=(p[0][i]+p[1][i]+p[2][i])/3

opt=np.max(c)

n_arms=10;
T=365
n_experiment=50

ts_rewards=[] #reward per gli esperimenti del ts algorithm
ucb1_rewards=[] #reward per gli esperimenti dell'algoritmo ucb1

totalRevenueTS=[]
totalRevenueUCB1=[]
dailyTS=0
dailyUCB1=0

cumRewardTS=0
cumRewardUCB1=0




nrClick=500



env = Enviroment(n_arms= n_arms, probabilities=c)
ts_learner= TS_Learner(n_arms=n_arms)
ucb1_learner= UCB1(n_arms= n_arms)
for t in range (0,T):
    
    #pull TS arm
    pulled_armTS=ts_learner.pull_arm() #prendo l'arm
    
    #pull UCB1 arm
    pulled_armUCB1=ucb1_learner.pull_arm()
    
    #I pull a new arm every day, and i propose the same price to the same person
    
    for c in range(0, nrClick):
        #Thompson Sampling Learner
        
        reward= env.round(pulled_armTS) #calcolo il reward
        cumRewardTS+=reward*(prezzi[pulled_armTS]/max_prezzo)
        dailyTS+=reward*prezzi[pulled_armTS]
        
        
        #totalRevenueTS=totalRevenueTS.append(prezzi[pulled_arm])
        #ts_learner.update(pulled_arm, reward)
        
        
        #UCB1 Learner 
        
        reward=env.round(pulled_armUCB1)
        cumRewardUCB1+=reward*(prezzi[pulled_armTS]/max_prezzo)
        dailyUCB1+=reward*prezzi[pulled_armUCB1]
        #totalRevenueUCB1=totalRevenueUCB1.append(prezzi[pulled_arm])
        #ucb1_learner.update(pulled_arm, reward)
    
    
    #make the average of the cumulative reward 
    totalRevenueTS.append(dailyTS)
    totalRevenueUCB1.append(dailyUCB1)
    ts_rewards.append(cumRewardTS/nrClick)
    ucb1_rewards.append(cumRewardUCB1/nrClick)
    #print(totalRevenueTS[t])
    #print(totalRevenueUCB1[t])
    ts_learner.update(pulled_armTS,cumRewardTS/nrClick)
    ucb1_learner.update(pulled_armUCB1, cumRewardUCB1/nrClick)
    dailyTS=0
    dailyUCB1=0
    cumRewardTS=0
    cumRewardUCB1=0


print(np.cumsum(totalRevenueTS)[364])
print(np.cumsum(totalRevenueUCB1)[364])



plt.figure(0)
plt.xlabel("t")
plt.ylabel("Revenue")
plt.plot(np.cumsum(totalRevenueTS, axis=0), 'r')
plt.plot(np.cumsum(totalRevenueUCB1, axis=0), 'b')
plt.legend(["TS", "UCB1"])



plt.figure(1)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(opt-ts_rewards, axis=0), 'r')
plt.plot(np.cumsum(opt-ucb1_rewards, axis=0), 'b')
plt.legend(["TS", "UCB1"])

plt.show()
