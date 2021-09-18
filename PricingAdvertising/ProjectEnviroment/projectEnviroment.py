# -*- coding: utf-8 -*-
"""
Spyder Editor
"""


import json
import numpy as np
import math
import os 

class contextEnv():
    
    def __init__(self):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, 'pricing_config.json')
        with open(path) as json_file:
            data = json.load(json_file)
        self.name=data["product"][0]["name"]
        self.features=data["product"][0]["features"]
        self.classes=data["product"][0]["classes"]
        self.prices=data["product"][0]["prices"]
        self.probabilities=data["product"][0]["probabilities"]
        
    def nextTimeProbability(month):
        return(month/(2+month))
    
    def nrDailyClick(bid):
        meanNrClick = math.trunc(500*math.tanh(bid))
        nrClick= math.trunc(np.random.uniform(meanNrClick-0.05*meanNrClick, meanNrClick+0.05*meanNrClick,1))
        return nrClick
    
    def costPerClick(self, bid):
        s= np.random.uniform(bid-0.05*bid, bid+0.05*bid,1)
        return s
        
    def conversionRateFunctionMale(price):
        rate=0.0001*(price)^3-0.0072*(price)^2+0.0842*price +0.3294
        return rate
    
    def conversionRateFunctionFelameA(price):
        rate= 0.0001*(price)^3-0.0053*(price)^2+0.0988*price+0.1855
        return rate
    
    def conversionRateFunctionFemaleY(price):
        rate=0.0001*(price)^3-0.0074*(price)^2+0.0804*price+0.3241
        return rate
        

# with open('C:/Users/MorganaGiorgio/PricingAdvertising/ProjectEnviroment/pricing_config.json') as json_file:
#     data = json.load(json_file)

"""
p=(data["product"][0]["probabilities"])
print(p)
c=np.zeros(10)
for i in range(0,9):
    c[i]=(p[0][i]+p[1][i]+p[2][i])/3
print(c) 
print(np.max(c))
"""