# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:39:49 2021

@author: MorganaGiorgio
"""

import numpy as np 
import matplotlib.pyplot as plt
from Enviroment import *
from TS_Learner import *
from Greedy_Learner import *

n_arms=4  #numero di arms 
p = np.array([0.15, 0.1, 0.1, 0.35])#le probabilità delle rispettive arms
opt = p[3] # in teoria l'ultima arms è quella ottimale

T=300 #time horizon dell'esperimento

n_experiment = 1000 #ne settiamo così tanti per rimuovere il rumore
ts_rewards_per_experiment=[] #reward per gli esperimenti del ts algorithm
gr_rewards_per_experiment=[] #reward per gli esperimenti dell'algoritmo greedy

for e in range(0,n_experiment):
    env = Enviroment(n_arms= n_arms, probabilities=p)
    ts_learner= TS_Learner(n_arms=n_arms)
    gr_learner= Greedy_Learner(n_arms= n_arms)
    for t in range (0,T):
        #Thompson Sampling Learner
        pulled_arm=ts_learner.pull_arm() #prendo l'arm
        reward= env.round(pulled_arm) #calcolo il reward
        ts_learner.update(pulled_arm, reward)
        
        
        #Greedy Learner 
        pulled_arm=gr_learner.pull_arm()
        reward=env.round(pulled_arm)
        gr_learner.update(pulled_arm, reward)

    ts_rewards_per_experiment.append(ts_learner.collected_rewards)
    gr_rewards_per_experiment.append(gr_learner.collected_rewards)

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Regret")
plt.plot(np.cumsum(np.mean(opt-ts_rewards_per_experiment, axis=0)), 'r')
plt.plot(np.cumsum(np.mean(opt-gr_rewards_per_experiment, axis=0)), 'g')
plt.legend(["TS", "Greedy"])
plt.show()