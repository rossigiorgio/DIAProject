# -*- coding: utf-8 -*-


from ProjectEnviroment.projectEnviroment import *
import numpy as np
from pricingEnviroment.Enviroment import *
from pricingEnviroment.TS_Learner import *
from pricingEnviroment.UCB1 import *
#from Learner import Learner
import matplotlib.pyplot as plt
import math

context= contextEnv()
prezzi=context.prices

print(prezzi)

p=context.probabilities

fixedBid=1.0
costPerClick=1.0


optClass1=np.max(p[0])
optClass2=np.max(p[1])
optClass3=np.max(p[2])

max_prezzo= np.max(prezzi)


n_arms=10;
T=365
n_experiment=50



totalRevenueTS=[]


totalRevenueSingle=[]


#reward if for every class i use his optimal arm
dailyTS=0

dailyTSSingle=0


cumRewardTSC1=0
cumRewardTSC2=0
cumRewardTSC3=0
cumRewardSingle=0

nrClickPerClass=127


#for each class define a new enviroment with his respective probabilities
envClass1 = Enviroment(n_arms= n_arms, probabilities=p[0])
envClass2 = Enviroment(n_arms= n_arms, probabilities=p[1])
envClass3 = Enviroment(n_arms= n_arms, probabilities=p[2])

#for every class define 3 different TS learner
ts_learnerClass1= TS_Learner(n_arms=n_arms)
ts_learnerClass2= TS_Learner(n_arms=n_arms)
ts_learnerClass3= TS_Learner(n_arms=n_arms)

ts_learnerSingle =TS_Learner(n_arms=n_arms)



for t in range (0,T):
    
    
    #pull TS arms
    pulled_armC1=ts_learnerClass1.pull_arm() 
    pulled_armC2=ts_learnerClass2.pull_arm() 
    pulled_armC3=ts_learnerClass3.pull_arm() 
    pulled_armSingle=ts_learnerSingle.pull_arm()
    
    
    
    
    #I pull a new arm every day, and i can distinguish the three different
    #class, and propose to each of them a different class
    
    #Class 1
    
    for x in range(0, nrClickPerClass):
        #Getting reward from his best arm
        reward= envClass1.round(pulled_armC1) #calcolo il reward
        cumRewardTSC1+=reward*(prezzi[pulled_armC1]/max_prezzo)
        dailyTS+=reward*prezzi[pulled_armC1]-costPerClick
    
        #using single learner
 
        reward= envClass1.round(pulled_armSingle)    
        dailyTSSingle+= reward*prezzi[pulled_armSingle]-costPerClick
        cumRewardSingle+=reward*(prezzi[pulled_armSingle]/max_prezzo)
    
    
        
    #Class 2
    for x in range(0, nrClickPerClass):
        #Getting reward from his best arm
        reward= envClass2.round(pulled_armC2) #calcolo il reward
        cumRewardTSC2+=reward*(prezzi[pulled_armC2]/max_prezzo)
        dailyTS+=reward*prezzi[pulled_armC2]-costPerClick
               
               
        #using single learner
        
        reward= envClass2.round(pulled_armSingle)
        dailyTSSingle+= reward*prezzi[pulled_armSingle]-costPerClick
        cumRewardSingle+=reward*(prezzi[pulled_armSingle]/max_prezzo)
        
     
    #Class 3
    for x in range(0, nrClickPerClass):
        #Getting reward from his best arm
        reward= envClass3.round(pulled_armC3) #calcolo il reward
        cumRewardTSC3+=reward*(prezzi[pulled_armC3]/max_prezzo)
        dailyTS+=reward*prezzi[pulled_armC3]-costPerClick
                
        
        #using single learner
        
        reward= envClass3.round(pulled_armSingle)
        dailyTSSingle+= reward*prezzi[pulled_armSingle]-costPerClick
        cumRewardSingle+=reward*(prezzi[pulled_armSingle]/max_prezzo)
    
    
    #make the average of the cumulative reward 
    totalRevenueTS.append(dailyTS)
    totalRevenueSingle.append(dailyTSSingle)
    
    
    
    ts_learnerClass1.update(pulled_armC1,cumRewardTSC1/nrClickPerClass)
    ts_learnerClass2.update(pulled_armC2,cumRewardTSC2/nrClickPerClass)
    ts_learnerClass3.update(pulled_armC3,cumRewardTSC3/nrClickPerClass)
    ts_learnerSingle.update(pulled_armSingle, cumRewardSingle/(3*nrClickPerClass))
    
    
    
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
plt.legend(["OptimalArm", "Single Learner"])



plt.show()
