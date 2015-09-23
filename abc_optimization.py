#!/usr/bin/python3
# -*- coding: utf8 -*-
"""
@author: Daniel Araujo Miranda
Licence: Free and open source under the GPL license
"""

import numpy as np
import fitnessfunctions

#default Artificial Bee Colony (ABC) parameters:
default_parameters={
  #Control of the algorithm:
  'NP':24,               #number of colony size (employed bees+onlooker bees)
  'FoodNumber':12,       #NP/2,  #number of food sources equals the half of the colony size
  'limit':20,            #a food source which could not be improved through "limit" trials is abandoned by its employed bee
  'maxCycle':1000,       #number of cycles for foraging (stopping condition)

  #Variables specific to the problem:
  'minThreshold':1.01,   #target cost (stopping condition)
  'D':6,                 #number of parameters of the problem to be optimized (dimensions)
  'ub':np.ones(6)*8.,    #lower bounds of the parameters
  'lb':np.ones(6)*(-8.)  #upper bounds of the parameters
}

def abc(f,parameters):
  #f                        #fitnessfunctions.rosenbrock #cost function to be optimized
  #Foods [FoodNumber][D]    #Foods is the population of food sources. Each row of Foods matrix is a vector holding D parameters to be optimized. The number of rows of Foods matrix equals to the FoodNumber
  #ObjVal[FoodNumber]       #f is a vector holding objective function values associated with food sources 
  #Fitness[FoodNumber]      #fitness is a vector holding fitness (quality) values associated with food sources
  #trial[FoodNumber]        #trial is a vector holding trial numbers through which solutions can not be improved
  #prob[FoodNumber]         #prob is a vector holding probabilities of food sources (solutions) to be chosen
  #solution [D]             #New solution (neighbour) produced by v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) j is a randomly chosen parameter and k is a randomlu chosen solution different from i
  #ObjValSol                #Objective function value of new solution
  #FitnessSol               #Fitness value of new solution
  #neighbour, param2change  #param2change corrresponds to j, neighbour corresponds to k in equation v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij})
  #GlobalMin                #Optimum solution obtained by ABC algorithm
  #GlobalParams[D]          #Parameters of the optimum solution
  #GlobalMins[runtime]      #GlobalMins holds the GlobalMin of each run in multiple runs
  global default_parameters
  p=default_parameters.copy()
  p.update(parameters)
  NP=p['NP']
  FoodNumber=p['FoodNumber']
  limit=p['limit']
  maxCycle=p['maxCycle']
  minThreshold=p['minThreshold']
  D=p['D']
  ub=p['ub']
  lb=p['lb']
  
  GlobalMins=[];
  
  #All food sources are initialized
  #Variables are initialized in the range from lb to ub
  Range=(ub-lb).reshape((1,-1)).repeat(FoodNumber,axis=0)
  Lower=lb.reshape((1,-1)).repeat(FoodNumber,axis=0)
  Foods=np.random.rand(FoodNumber,D)*Range+Lower
  del Range,Lower
  #initializing cost calculations
  ObjVal=np.array([f(i) for i in Foods])
  #reset trial counters
  trial=np.zeros(FoodNumber)
  
  #The best food source is memorized
  BestInd=np.where(ObjVal==np.min(ObjVal))[0][0]
  GlobalMin=ObjVal[BestInd]
  GlobalParams=Foods[BestInd].copy()
  GlobalMins.append(GlobalMin)
  
  iteration=0
  while (iteration<maxCycle):
    iteration=iteration+1
    ##################EMPLOYED BEE PHASE##################
    #The parameter to be changed is determined randomly
    Param2Change=np.random.randint(0,D,FoodNumber)
    #A randomly chosen solution is used in producing a mutant solution of the solution i
    aux=np.arange(FoodNumber,dtype='int')  
    neighbour=(np.random.randint(1,FoodNumber,FoodNumber)+aux)%FoodNumber
    new_solution=Foods.copy()
    new_solution[(aux,Param2Change)]=Foods[(aux,Param2Change)]+(np.random.rand(FoodNumber)*2-1)*(Foods[(aux,Param2Change)]-Foods[(neighbour,Param2Change)])
    del aux
    #if generated parameter value is out of boundaries, it is shifted onto the boundaries
    new_solution=np.maximum(new_solution,lb)
    new_solution=np.minimum(new_solution,ub)
    #evaluate new solution
    ObjValSol=np.array([f(x) for x in new_solution])
    #a greedy selection is applied between the current solution i and its mutant
    aux1=np.where(ObjValSol<ObjVal)
    aux2=np.where(ObjValSol>=ObjVal)
    #If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i
    Foods[aux1]=new_solution[aux1]
    ObjVal[aux1]=ObjValSol[aux1]
    trial[aux1]=0
    #if the solution i can not be improved, increase its trial counter
    trial[aux2]=trial[aux2]+1
    del aux1,aux2,ObjValSol,new_solution,neighbour
    
    ##################CALCULATE PROBABILITIES##################
    # A food source is chosen with the probability which is proportional to its quality
    #Different schemes can be used to calculate the probability values
    #For example prob(i)=fitness(i)/sum(fitness)
    #or in a way used in the metot below prob(i)=a*fitness(i)/max(fitness)+b
    #probability values are calculated by using fitness values and normalized by dividing maximum fitness value
    prob=(0.9*ObjVal/np.max(ObjVal))+0.1;
    ##################ONLOOKER BEE PHASE##################
    aux=np.where(prob>np.random.rand(FoodNumber))[0] #chosen food sources by probability
    n_aux=len(aux) #number of randomly chosen food sources
    #The parameter to be changed is determined randomly
    Param2Change=np.random.randint(0,D,n_aux)
    #A randomly chosen solution is used in producing a mutant solution of the solution i
    #Randomly selected solution must be different from the solution i
    neighbour=(np.random.randint(1,FoodNumber,n_aux)+aux)%FoodNumber
    new_solution=Foods.copy()
    new_solution[(aux,Param2Change)]=Foods[(aux,Param2Change)]+(np.random.rand(n_aux)*2-1)*(Foods[(aux,Param2Change)]-Foods[(neighbour,Param2Change)])
    del aux,n_aux
    #if generated parameter value is out of boundaries, it is shifted onto the boundaries
    new_solution=np.maximum(new_solution,lb)
    new_solution=np.minimum(new_solution,ub)
    #evaluate new solution
    ObjValSol=np.array([f(x) for x in new_solution])
    #a greedy selection is applied between the current solution i and its mutant
    aux1=np.where(ObjValSol<ObjVal)
    aux2=np.where(ObjValSol>=ObjVal)
    #If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i
    Foods[aux1]=new_solution[aux1]
    ObjVal[aux1]=ObjValSol[aux1]
    trial[aux1]=0
    #if the solution i can not be improved, increase its trial counter
    trial[aux2]=trial[aux2]+1
    del aux1,aux2,ObjValSol,new_solution,neighbour
  
    #The best food source is memorized
    BestInd=np.where(ObjVal==np.min(ObjVal))[0][0]
    GlobalMin=ObjVal[BestInd]
    GlobalParams=Foods[BestInd].copy()
    GlobalMins.append(GlobalMin)
  
    #stop if threshold reached
    if GlobalMin <= minThreshold:
      print("Target threshold reached!")
      break
  
    min_before_scout=np.min(ObjVal)
    ################## Scout Bee Phase ##################
    #determine the food sources whose trial counter exceeds the "limit" value. 
    #In Basic ABC, only one scout is allowed to occur in each cycle
    aux=np.where(trial==np.max(trial))[0][-1]
    if(trial[aux]>limit):
      new_food=np.random.rand(D)*(ub-lb)+lb
      new_cost=f(new_food)
      if(new_cost<ObjVal[aux]): #fix for scout bees destroying the global minimum
      #if(True):
        Foods[aux]=new_food.copy()
        ObjVal[aux]=new_cost
        #print('Scount replaced food source',aux)
      #else:
      #  print('Scount tried to replace food source',aux,'and failed.')
      trial[aux]=0
      
      #break
    del aux
    if(iteration%100==0):
      print('Iteration',iteration,', global minumum=',GlobalMin);
    min_after_scout=np.min(ObjVal)
    if(min_before_scout<min_after_scout):
      print("WARNING: Scout replaced optimum solution.")
      print('min_before:',min_before_scout,'min_after:',min_after_scout)
    #end of algorithm
  
  print('Iteration',iteration,', global minumum=',GlobalMin);
  print('Global Parameters:')
  print(GlobalParams)
  return GlobalMins

#quick test:
cost_history=abc(fitnessfunctions.rastrigin,
                 {'minThreshold':0.001,
                 'maxCycle':500,
                 'D':6,
                 'lb':-8.*np.ones(6),
                 'ub': 8.*np.ones(6)})

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax1=fig.add_subplot(211)
#ax1.plot(np.log10(cost_history))
ax1.plot(cost_history)
ax2=fig.add_subplot(212)
ax2.plot(np.log10(cost_history))
#ax2.set_ylim((-2,1))
plt.show()
