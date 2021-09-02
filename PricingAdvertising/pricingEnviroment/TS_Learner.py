# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:29:39 2021

@author: MorganaGiorgio
"""

from pricingEnviroment.Learner import *

class TS_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        #estendo la classe learner precedente con il parametro dela beta distribution
        self.beta_parameters = np.ones((n_arms, 2)) #i parametri della beta distribution sono due
        #alpha e beta ed entrambi inizialmente sono inizializzati a 1, per quest creo un nuovo array 
        #bidimensionale di 1 con lunghezza = al numero di arms
        
    def pull_arm(self):
        #per ogni arm quello che faccio è di prendere un sample dalla beta distribution per ogni arm 
        #ad ogni iterazione. L'arm che verrà pullata sarà quella con il sample con il valore maggiore
        #in questo caso infatti per ogni arm tramite la funzione np.random.beta creo dei sample e con la
        #funzione argmax seleziono l'indice di quello con il valore maggiore
        idx=np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1]))
        return idx
    
    def update(self, pulled_arm, reward):
        self.t+=1; #incremento il tempo
        self.update_observations(pulled_arm, reward) #faccio l'update delle observation
        self.beta_parameters[pulled_arm, 0]= self.beta_parameters[pulled_arm,0]+reward
        self.beta_parameters[pulled_arm, 1]= self.beta_parameters[pulled_arm,1] +(1.0 - reward)