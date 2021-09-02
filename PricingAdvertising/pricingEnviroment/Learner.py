# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:01:05 2021

@author: MorganaGiorgio
"""

import numpy as np

class Learner:
    def __init__(self, n_arms):
        self.n_arms=n_arms
        self.t=0 #current round value, cioè il tempo t, inizializzato a 0
        #questa variabile è una lista di liste e la dimensione della lista esterna
        #è uguale al numero di arms mentre la dimensione della lista interna è data
        #da quante volte ho pullato una determinata arm 
        self.rewards_per_arm= x = [ [] for i in range (n_arms)]
        self. collected_rewards = np.array([])
    
    def update_observations(self, pulled_arm, reward):
        #quando ho pullato l'arm inserisco i valori ottenuti
        self.rewards_per_arm[pulled_arm].append(reward) 
        self.collected_rewards= np.append(self.collected_rewards, reward)
    
    