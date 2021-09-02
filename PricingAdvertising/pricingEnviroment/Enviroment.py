# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:58:22 2021

@author: MorganaGiorgio
"""
import numpy as np

class Enviroment():
    def __init__(self, n_arms, probabilities):
        self.narms=n_arms
        self.probabilities = probabilities
    
    def round(self, pulled_arm):
        reward= np.random.binomial(1, self.probabilities[pulled_arm])
        return reward
    
    