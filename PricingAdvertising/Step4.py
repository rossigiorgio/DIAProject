# -*- coding: utf-8 -*-


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

p=context.probabilities



optClass1=np.max(p[0])
optClass2=np.max(p[1])
optClass3=np.max(p[2])



n_arms=10;
T=365
n_experiment=50

ts_rewardsC1=[] #reward per gli esperimenti del ts algorithm
ts_rewardsC2=[] #reward per gli esperimenti del ts algorithm
ts_rewardsC3=[] #reward per gli esperimenti del ts algorithm

totalRevenueTS=[]

totalRevenueC1Best=[]
totalRevenueC2Best=[]
totalRevenueC3Best=[]


#reward if for every class i use his optimal arm
dailyTS=0

dailyC1Best=0
dailyC2Best=0
dailyC3Best=0


rewardAlwaysC1Best=0
rewardAlwaysC2Best=0
rewardAlwaysC3Best=0


cumRewardTSC1=0
cumRewardTSC2=0
cumRewardTSC3=0

nrClickPerClass=167


#for each class define a new enviroment with his respective probabilities
envClass1 = Enviroment(n_arms= n_arms, probabilities=p[0])
envClass2 = Enviroment(n_arms= n_arms, probabilities=p[1])
envClass3 = Enviroment(n_arms= n_arms, probabilities=p[2])

#for every class define 3 different TS learner
ts_learnerClass1= TS_Learner(n_arms=n_arms)
ts_learnerClass2= TS_Learner(n_arms=n_arms)
ts_learnerClass3= TS_Learner(n_arms=n_arms)



for t in range (0,T):
    
    #pull TS arms
    pulled_armC1=ts_learnerClass1.pull_arm() 
    pulled_armC2=ts_learnerClass2.pull_arm() 
    pulled_armC3=ts_learnerClass2.pull_arm() 
   
    
    #I pull a new arm every day, and i can distinguish the three different
    #class, and propose to each of them a different class
    
    #Class 1
    
    for x in range(0, nrClickPerClass):
        #Getting reward from his best arm
        reward= envClass1.round(pulled_armC1) #calcolo il reward
        cumRewardTSC1+=reward
        dailyTS+=reward*prezzi[pulled_armC1]
        dailyC1Best+=reward*prezzi[pulled_armC1]
        
        #Getting reward if i use the price from the arm pulled by C2
        reward= envClass1.round(pulled_armC2) #getting reward from the optimal price for C2
        dailyC2Best+=reward*prezzi[pulled_armC2]
        
        
        #Getting reward if i use the price from the arm pulled by C3
        reward= envClass1.round(pulled_armC3) #getting reward from the optimal price for C3
        dailyC3Best+= reward*prezzi[pulled_armC3]
    
        
    #Class 2
    for x in range(0, nrClickPerClass):
        #Getting reward from his best arm
        reward= envClass2.round(pulled_armC2) #calcolo il reward
        cumRewardTSC2+=reward
        dailyTS+=reward*prezzi[pulled_armC2]
        dailyC2Best+= reward*prezzi[pulled_armC2]
        
        #Getting reward if i use the price from the arm pulled by C1
        reward= envClass2.round(pulled_armC1) #getting reward from the optimal price for C2
        dailyC1Best+=reward*prezzi[pulled_armC1]
        
        
        #Getting reward if i use the price from the arm pulled by C3
        reward= envClass2.round(pulled_armC3) #getting reward from the optimal price for C3
        dailyC3Best += reward*prezzi[pulled_armC3]
        
        
        
     
    #Class 3
    for x in range(0, nrClickPerClass):
        #Getting reward from his best arm
        reward= envClass3.round(pulled_armC3) #calcolo il reward
        cumRewardTSC3+=reward
        dailyTS+=reward*prezzi[pulled_armC3]
        dailyC3Best+=reward*prezzi[pulled_armC3]
        
        
        #Getting reward if i use the price from the arm pulled by C1
        reward= envClass3.round(pulled_armC1) #getting reward from the optimal price for C2
        dailyC1Best+=reward*prezzi[pulled_armC1]
        
        
        #Getting reward if i use the price from the arm pulled by C3
        reward= envClass3.round(pulled_armC2) #getting reward from the optimal price for C3
        dailyC2Best += reward*prezzi[pulled_armC2]
        

    
    
    #make the average of the cumulative reward 
    totalRevenueTS.append(dailyTS)
    totalRevenueC1Best.append(dailyC1Best)
    totalRevenueC2Best.append(dailyC2Best)
    totalRevenueC3Best.append(dailyC3Best)
    
    ts_rewardsC1.append(cumRewardTSC1/nrClickPerClass)  
    ts_rewardsC2.append(cumRewardTSC2/nrClickPerClass)  
    ts_rewardsC3.append(cumRewardTSC3/nrClickPerClass)  
    
    ts_learnerClass1.update(pulled_armC1,cumRewardTSC1/nrClickPerClass)
    ts_learnerClass2.update(pulled_armC2,cumRewardTSC2/nrClickPerClass)
    ts_learnerClass3.update(pulled_armC3,cumRewardTSC3/nrClickPerClass)
    
    dailyTS=0
    
    cumRewardTSC1=0
    cumRewardTSC2=0
    cumRewardTSC3=0
    
    dailyC1Best=0
    dailyC2Best=0
    dailyC3Best=0

print(np.cumsum(totalRevenueTS, axis=0)[364])
print(np.cumsum(totalRevenueC1Best, axis=0)[364])
print(np.cumsum(totalRevenueC2Best, axis=0)[364])
print(np.cumsum(totalRevenueC3Best, axis=0)[364])


plt.figure(0)
plt.xlabel("t")
plt.ylabel("Revenue")
plt.plot(np.cumsum(totalRevenueTS, axis=0), 'r')
plt.plot(np.cumsum(totalRevenueC1Best, axis=0), 'b')
plt.legend(["OptimalArm", "All C1"])

plt.show()

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Revenue")
plt.plot(np.cumsum(totalRevenueTS, axis=0), 'r')
plt.plot(np.cumsum(totalRevenueC2Best, axis=0), 'b')
plt.legend(["OptimalArm", "All C2"])

plt.show()

plt.figure(0)
plt.xlabel("t")
plt.ylabel("Revenue")
plt.plot(np.cumsum(totalRevenueTS, axis=0), 'r')
plt.plot(np.cumsum(totalRevenueC3Best, axis=0), 'b')
plt.legend(["OptimalArm", "All C3"])

plt.show()
