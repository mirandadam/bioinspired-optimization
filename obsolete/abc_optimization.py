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
  #Algorithm tuning:
  'NB':24,                 #Number of bees (employed bees plus onlooker bees)
  'NF':12,                 #Number of food sources. Default is NB/2.
  'abandon_threshold':20,  #Number of consecutive improvement trials that a food source undergoes before being abandoned
  'maxiter':1000,          #Stopping condition. Maximum number of foraging cycles to try.

  #Problem parameters:
  'target_cost':0.5,     #Stopping condition. If a solution reaches this cost or lower the algorithm halts.
  'dimensions':6,        #Number of parameters of the problem to be optimized (dimensions)
  'ub':np.ones(6)*8.,    #lower bounds of the parameters
  'lb':np.ones(6)*(-8.)  #upper bounds of the parameters
}


def abc(f=fitnessfunctions.rosenbrock_m, parameters=default_parameters):
  global default_parameters
  p=default_parameters.copy()
  p.update(parameters)
  NB=p['NB'] 
  NF=p['NF'] 
  abandon_threshold=p['abandon_threshold']
  maxiter=p['maxiter']
  target_cost=p['target_cost']
  dimensions=p['dimensions']
  ub=p['ub']
  lb=p['lb']
  del p, parameters
  
  cost_history=[];
  
  #All food sources are initialized
  #Variables are initialized in the range from lb to ub
  parameter_range=(ub-lb).reshape((1,-1)).repeat(NF,axis=0)
  parameter_lower_bound=lb.reshape((1,-1)).repeat(NF,axis=0)
  foods=np.random.rand(NF,dimensions)*parameter_range+parameter_lower_bound
  del parameter_range, parameter_lower_bound
  #initializing cost calculations
  food_costs=f(foods)#np.array([f(i) for i in foods])
  #reset trial counters
  trials=np.zeros(NF,dtype='uint')
  
  #The best food source is memorized
  min_cost_index=np.where(food_costs==np.min(food_costs))[0][0]
  min_cost=food_costs[min_cost_index]
  min_cost_params=foods[min_cost_index].copy()
  cost_history.append(min_cost)
  
  iteration=0
  while (iteration<maxiter):
    iteration=iteration+1
    ##################EMPLOYED BEE PHASE##################
    #The parameter to be changed is determined randomly
    parameters_to_change=np.random.randint(0,dimensions,NF)
    #A randomly chosen solution is used in producing a mutant solution of the solution i
    aux=np.arange(NF,dtype='int')  
    neighbours=(np.random.randint(1,NF,NF)+aux)%NF
    new_foods=foods.copy()
    new_foods[(aux,parameters_to_change)]=foods[(aux,parameters_to_change)]+(np.random.rand(NF)*2-1)*(foods[(aux,parameters_to_change)]-foods[(neighbours,parameters_to_change)])
    del aux
    del neighbours, parameters_to_change
    #if generated parameter value is out of boundaries, it is shifted onto the boundaries
    new_foods=np.maximum(new_foods,lb)
    new_foods=np.minimum(new_foods,ub)
    #evaluate new solution
    new_food_costs=f(new_foods)#np.array([f(x) for x in new_foods])
    #a greedy selection is applied between the current solution i and its mutant
    aux1=np.where(new_food_costs<food_costs)
    aux2=np.where(new_food_costs>=food_costs)
    #If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i
    foods[aux1]=new_foods[aux1]
    food_costs[aux1]=new_food_costs[aux1]
    trials[aux1]=0
    #if the solution i can not be improved, increase its trial counter
    trials[aux2]=trials[aux2]+1
    del aux1,aux2,new_food_costs,new_foods
    
    ##################CALCULATE PROBABILITIES##################
    # A food source is chosen with the probability which is proportional to its quality
    #Different schemes can be used to calculate the probability values
    #For example prob(i)=fitness(i)/sum(fitness)
    #or in a way used in the metot below prob(i)=a*fitness(i)/max(fitness)+b
    #probability values are calculated by using fitness values and normalized by dividing maximum fitness value
    prob=(0.9*food_costs/np.max(food_costs))+0.1; #the higher the cost, the higher the probability for change
    ##################ONLOOKER BEE PHASE##################
    aux=np.where(prob>np.random.rand(NF))[0] #chosen food sources by probability
    n_aux=len(aux) #number of randomly chosen food sources
    #The parameter to be changed is determined randomly
    parameters_to_change=np.random.randint(0,dimensions,n_aux)
    #A randomly chosen solution is used in producing a mutant solution of the solution i
    #Randomly selected solution must be different from the solution i
    neighbours=(np.random.randint(1,NF,n_aux)+aux)%NF
    new_foods=foods.copy()
    new_foods[(aux,parameters_to_change)]=foods[(aux,parameters_to_change)]+(np.random.rand(n_aux)*2-1)*(foods[(aux,parameters_to_change)]-foods[(neighbours,parameters_to_change)])
    del aux,n_aux
    #if generated parameter value is out of boundaries, it is shifted onto the boundaries
    new_foods=np.maximum(new_foods,lb)
    new_foods=np.minimum(new_foods,ub)
    #evaluate new solution
    new_food_costs=f(new_foods)#np.array([f(x) for x in new_foods])
    #a greedy selection is applied between the current solution i and its mutant
    aux1=np.where(new_food_costs<food_costs)
    aux2=np.where(new_food_costs>=food_costs)
    #If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i
    foods[aux1]=new_foods[aux1]
    food_costs[aux1]=new_food_costs[aux1]
    trials[aux1]=0
    #if the solution i can not be improved, increase its trial counter
    trials[aux2]=trials[aux2]+1
    del aux1,aux2,new_food_costs,new_foods,neighbours
  
    #The best food source is memorized
    min_cost_index=np.where(food_costs==np.min(food_costs))[0][0]
    min_cost=food_costs[min_cost_index]
    min_cost_params=foods[min_cost_index].copy()
    cost_history.append(min_cost)
  
    #stop if target_cost reached
    if min_cost <= target_cost:
      #print("Target cost reached!")
      break
  
    min_before_scout=np.min(food_costs)
    ################## Scout Bee Phase ##################
    #determine the food sources whose trial counter exceeds the "abandon_threshold" value. 
    #In Basic ABC, only one scout is allowed to occur in each cycle
    aux=np.where(trials==np.max(trials))[0][-1]
    if(trials[aux]>abandon_threshold):
      new_food=np.random.rand(dimensions)*(ub-lb)+lb
      new_cost=f(np.array([new_food]))[0]
      #if(new_cost<food_costs[aux]): #fix for scout bees destroying the global minimum
      if(True):
        foods[aux]=new_food.copy()
        food_costs[aux]=new_cost
        #print('Scount replaced food source',aux)
      #else:
      #  print('Scount tried to replace food source',aux,'and failed.')
      trials[aux]=0
      
      #break
    del aux
    if(iteration%100==0):
      #print('Iteration',iteration,', global minumum=',min_cost);
      pass
    min_after_scout=np.min(food_costs)
    if(min_before_scout<min_after_scout):
      #print("WARNING: Scout replaced optimum solution.")
      #print('min_before:',min_before_scout,'min_after:',min_after_scout)
      pass
    #end of algorithm
  
  #print('Iteration',iteration,', final global minumum=',min_cost);
  #print('Global Parameters:')
  #print(min_cost_params)
  return cost_history

#quick test:
#cost_history=abc(fitnessfunctions.rosenbrock_m,
#cost_history=abc(fitnessfunctions.schwefel_m,
cost_history=abc(fitnessfunctions.sphere_m,
                 {'target_cost':0.01,
                  'abandon_threshold':100,
                  'maxiter':500,
                  'dimensions':6,
                  'lb':-8.*np.ones(6),
                  'ub': 8.*np.ones(6)})

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax1=fig.add_subplot(211)
#ax1.plot(np.log10(cost_history))
ax1.plot(cost_history)
ax1.set_ylabel("$mincost$")
ax2=fig.add_subplot(212)
ax2.plot(np.log10(cost_history))
ax2.set_ylabel("$log_{10}(mincost)$")
ax2.set_xlabel("$iteration$")
#ax2.set_ylim((-2,1))
plt.show()
#'''