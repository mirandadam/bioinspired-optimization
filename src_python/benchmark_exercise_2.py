#!/usr/bin/python3
# -*- coding: utf8 -*-

#Copyright (C) 2015 Daniel Araujo Miranda
#
#This program is free software; you can redistribute it and/or
#modify it under the terms of the GNU General Public License
#as published by the Free Software Foundation; either version 2
#of the License, or (at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#@author: Daniel Araujo Miranda

import numpy as np
from time import time
import pickle #for storing and retrieving the raw results
import bio
import fitnessfunctions

functions={i:fitnessfunctions.all_functions[i] for i in ['Michalewicz','Rosenbrock','RotatedHyperEllipsoid','Schwefel','Rastrigin','Sphere']}
algorithms={i:bio.all_algorithms[i] for i in ['FA','DE']}

swarm_sizes=[20]
dimensions=[2,6,12]
number_of_repetitions=32

fa_parameters=dict(
 alpha=0.8,
 delta=0.99,
 exponent=0.85, #gamma
 beta0=1
)

de_parameters=dict(
f=1.2,
c=0.95
)

arg_dict=dict(FA=fa_parameters, DE=de_parameters)

## calculating the raw data: ##
'''
results=[]
t0=time()
for i in dimensions:
  for j in swarm_sizes:
    for f in functions.items():
      f_name=f[1].name
      f_eval=f[1].evaluate
      lb,ub=f[1].default_bounds(i)
      ymin,xmin=f[1].default_minimum(i) #target values
      for a in algorithms.items():
        a_name=a[1].name
        args=arg_dict[a_name]
        args['n']=j
        print (i,j,f[0],a[0])
        for k in range(number_of_repetitions):
          o=a[1](f_eval,i,lb,ub,maxiter=1000,**args)
          y,x=o.run()
          cost_delta=abs(y-ymin)
          solution_delta=((x-xmin)**2).sum()**0.5
          #dimensions, swarm size, cost function name, algorithm name,
          # index of repetition, final cost, theoretical minimum cost
          # difference between actual cost and theorietical minimum,
          # best solution found, theoretical best solution,
          # norm of the distance between the ideal solution and the solution fount
          results.append([i,j,f[0],a[0],k,y,ymin,cost_delta,x,xmin,solution_delta])

t1=time()
print('time elapsed:',t1-t0,'seconds.')

f=open("exercise_2_raw_data.pickle",'wb')
pickle.dump(results,f)
f.close()
#'''

## Outputting the results: ##
f=open("exercise_2_raw_data.pickle",'rb')
results=pickle.load(f)
f.close()

tables=sorted(list(set([(i[2],i[3]) for i in results])),reverse=True)
for t in range(len(tables)):
  print('Table '+str(t+1)+'. Convergency of the '+tables[t][1]+' algorithm applied to the '+tables[t][0]+' function.')
  print('\t\tMean\tMedian\tMinimum\tStandard deviation\tConvergence rate (%)')
  if(tables[t][0]=='Rosenbrock'):
    tolerance=1e-1
  else:
    tolerance=1e-2
  for s in swarm_sizes:
    print('S='+str(s),end='')
    for d in dimensions:
      costs=np.array([[i[5],i[6]] for i in results if (i[0],i[1],i[2],i[3])==(d,s,tables[t][0],tables[t][1])])
      mean_cost=np.mean(costs[:,0])
      median_cost=np.median(costs[:,0])
      min_cost=np.min(costs[:,0])
      stddev_cost=np.std(costs[:,0])
      success_rate=np.sum(np.abs(costs[:,0]-costs[:,1])<tolerance)/32
      print('\t N='+str(d)+'\t'+str(mean_cost)+'\t'+str(median_cost)+'\t'+str(min_cost)+'\t'+str(stddev_cost)+'\t'+str(round(success_rate*100,2)))
  for d in dimensions:
    costs=np.array([[i[5],i[6]] for i in results if (i[0],i[2],i[3])==(d,tables[t][0],tables[t][1])])
    solutions=np.array([[i[8],i[9]] for i in results if (i[0],i[2],i[3])==(d,tables[t][0],tables[t][1])])
    bestindex=np.argmin(costs[:,0])
    besty=costs[bestindex,0]
    bestx=solutions[bestindex,0]
    print("Best solution found for N="+str(d)+':',besty,"at point",bestx)
    print("  Theoretical best:",costs[bestindex,1],"at point",solutions[bestindex,1])
