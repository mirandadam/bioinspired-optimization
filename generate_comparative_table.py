#!/usr/bin/python3
# -*- coding: utf8 -*-
"""
@author: Daniel Araujo Miranda
Licence: Free and open source under the GPL license
"""

import os
import numpy as np
import fitnessfunctions
import pso_optimization
import abc_optimization

from time import time

functions=[
  [fitnessfunctions.sphere_m,'sphere',[-8,8],0.01],
  [fitnessfunctions.quadric_m,'quadric',[-8,8],0.01],
  [fitnessfunctions.rastrigin_m,'rastrigin',[-8,8],0.01],
  [fitnessfunctions.rosenbrock_m,'rosenbrock',[-8,8],0.1],
  [fitnessfunctions.schwefel_m,'schwefel',[-500,500],0.01],
  [fitnessfunctions.ackley_m,'ackley',[-32,32],0.01],
  [fitnessfunctions.michalewicz_m,'michalewicz',[0,np.pi],-100]
  ]

swarm_sizes=[10,15,20]
dimensions=[6,10,14,18,22]
number_of_repetitions=32

pso_parameters={
  'S':10,               #*number of particles
  'dimensions':6,       #*number of dimensions
  'maxiter':1000,        #maximum number of iterations
  'w0':0.9,              #initial weight
  'wf':0.1,              #final weight
  'c1':2.05,             #cognitive coefficient
  'c2':2.05,             #social coefficient
  'max_v':5,             #maximum velocity
  'ini_v':5/3.,#max_v/3  #initial velocity
  'x_max':8,            #*range domain for fitness function - inferior limit (assumes the same for all dimensions)
  'x_min':-8,           #*range domain for fitness function - superior limit (assumes the same for all dimensions)
  'target_cost':-1000   #NÃO utilizar como critério de parada.
}

abc_parameters={
  'NB':24,                #*Number of bees (employed bees plus onlooker bees)
  'NF':12,                #*Number of food sources. Default is NB/2.
  'abandon_threshold':20,  #Number of consecutive improvement trials that a food source undergoes before being abandoned
  'maxiter':1000,          #Stopping condition. Maximum number of foraging cycles to try.

  #Problem parameters:
  'target_cost':-1000,    #NÃO utilizar como critério de parada
  'dimensions':6,         #*Number of parameters of the problem to be optimized (dimensions)
  'ub':np.ones(6)*8.,     #*lower bounds of the parameters
  'lb':np.ones(6)*(-8.)   #*upper bounds of the parameters
}

t0=time()
datapoints=[]
for d in dimensions:
  for s in swarm_sizes:
    for fi in range(len(functions)):
      func=functions[fi]
      f=func[0]
      target=func[3]
      pso_parameters['S']=s
      pso_parameters['dimensions']=d
      pso_parameters['x_min']=func[2][0]
      pso_parameters['x_max']=func[2][1]
      abc_parameters['NB']=s*2
      abc_parameters['NF']=s
      abc_parameters['dimensions']=d
      abc_parameters['lb']=func[2][0]*np.ones(d)
      abc_parameters['ub']=func[2][1]*np.ones(d)
      print('d=',d,'s=',s,'f=',func[1])
      for r in range(number_of_repetitions):
        cost_pso=pso_optimization.pso(f,pso_parameters)
        datapoints.append([fi,0,r,d,s,cost_pso[-1],np.min(cost_pso)])
      for r in range(number_of_repetitions):
        cost_abc=abc_optimization.abc(f,abc_parameters)
        datapoints.append([fi,1,r,d,s,cost_abc[-1],np.min(cost_abc)])
t1=time()
print(t1-t0)

import pickle
f=open("datapoints.pickle",'wb')
pickle.dump(datapoints,f)
f.close()
#f=open("datapoints.pickle",'rb')
#datapoints=pickle.load(f)
#f.close()
dp=np.array(datapoints)

f=open("tables.txt",'w')

for fi in range(len(functions)):
  func=functions[fi]
  aux=os.linesep+'Algoritmo PSO. Resultados de convergência para a função '+func[1]+' ('+str(number_of_repetitions)+' repetições)'+os.linesep
  aux+='S\tN\tMédia\tMediana\tMínimo\tDesvio Padrão\tgoals/'+str(number_of_repetitions)
  f.write(aux+os.linesep)
  for s in swarm_sizes:
    for d in dimensions:
      temp=dp[np.where((dp[:,0]==fi) * (dp[:,1]==0) * (dp[:,3]==d) * (dp[:,4]==s))][:,5].copy()
      run_average=np.mean(temp)
      run_median=np.median(temp)
      run_minimum=np.min(temp)
      run_stddev=np.std(temp)
      goal=func[3]
      run_goal_fraction=np.sum(temp<=goal)/number_of_repetitions
      aux=[s,d,run_average,run_median,run_minimum,run_stddev,run_goal_fraction]
      run_output='\t'.join(list(map(str,aux)))
      #print(run_output)
      f.write(run_output+os.linesep)
  aux=os.linesep+'Algoritmo ABC. Resultados de convergência para a função '+func[1]+' ('+str(number_of_repetitions)+' repetições)'+os.linesep
  aux+='S\tN\tMédia\tMediana\tMínimo final\tDesvio Padrão\tgoals/'+str(number_of_repetitions)+'\tMinimo historico\tgoals/'+str(number_of_repetitions)
  f.write(aux+os.linesep)
  for s in swarm_sizes:
    for d in dimensions:
      temp=dp[np.where((dp[:,0]==fi) * (dp[:,1]==1) * (dp[:,3]==d) * (dp[:,4]==s))][:,5].copy()
      run_average=np.mean(temp)
      run_median=np.median(temp)
      run_minimum=np.min(temp)
      run_stddev=np.std(temp)
      goal=func[3]
      run_goal_fraction=np.sum(temp<=goal)/number_of_repetitions
      temp=dp[np.where((dp[:,0]==fi) * (dp[:,1]==1) * (dp[:,3]==d) * (dp[:,4]==s))][:,6].copy()
      run_historical_minimum=np.min(temp)
      run_historical_goal_fraction=np.sum(temp<=goal)/number_of_repetitions
      aux=[s,d,run_average,run_median,run_minimum,run_stddev,run_goal_fraction,run_historical_minimum,run_historical_goal_fraction]
      run_output='\t'.join(list(map(str,aux)))
      f.write(run_output+os.linesep)
      #print(run_output)

f.close()
  
