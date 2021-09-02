# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 16:01:15 2021

@author: MorganaGiorgio
"""
from pricingEnviroment.Enviroment import Enviroment
from pricingEnviroment.Learner import Learner
import numpy as np


class UCB1(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means= np.zeros(n_arms) #Xa
        self.confidence = np.zeros(n_arms) #extra term (quello sotto radice)
    
    def pull_arm(self): #pull the arm that maximixe the bound (see the formula)
        if self.t< self.n_arms:
            arm=self.t
        else:  
            upper_bound=self.empirical_means+self.confidence
            arm=np.random.choice(np.where(upper_bound==upper_bound.max())[0])
        return arm
         
    def update(self, pulled_arm, reward):
        self.t +=1        
        #self.collected_rewards= np.append(self.collected_rewards, reward)
        self.empirical_means[pulled_arm]=(self.empirical_means[pulled_arm]*(self.t-1)+reward)/self.t
        for a in range (self.n_arms):
            number_pulled= max(1, len(self.rewards_per_arm[a]))
            self.confidence[a]=(2*np.log(self.t)/number_pulled)**0.5
            
        self.update_observations(pulled_arm, reward)
            
"""         
if __name__ == '__main__':
    p=np.array([0.1, 0.8, 0.3])
    n_arms=len(p)
    env = Enviroment(n_arms=n_arms, probabilities=p)
    T=100
    ucb_learner= UCB1(n_arms)
    for _ in range (T):
        pulled_arm=ucb_learner.pull_arm()
        print(f"{pulled_arm=}")
        reward= env.round(pulled_arm)
        ucb_learner.update_observations(pulled_arm, reward)
    
 """   
    
    