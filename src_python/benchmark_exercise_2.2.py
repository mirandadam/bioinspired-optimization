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
algorithms={i:bio.all_algorithms[i] for i in ['FA_OBL','FA_CP','DE_OBL','DE_CP']}

swarm_sizes=[20]
dimensions=[12]
number_of_repetitions=32

fa_parameters=dict(
  alpha0=0.8,
  delta=0.99,
  exponent=0.85, #gamma
  beta0=1
  )

de_parameters=dict(
  f=1.2,
  c=0.95
  )

obl_parameters=dict(
  obl_iteration_threshold=40,  #number of iterations without improvement to trigger OBL
  obl_randomness=0.1,          #randomness to aply
  obl_probability=0.5          #probability of a coordinate to be flipped by the OBL
  )

cp_parameters=dict(
  cp_confidence_in_second=0.5  #Passive Congregation confidence in the second best individual
  )

fa_obl=fa_parameters.copy()
fa_obl.update(obl_parameters)
fa_cp=fa_parameters.copy()
fa_cp.update(cp_parameters)

de_obl=de_parameters.copy()
de_obl.update(obl_parameters)
de_cp=de_parameters.copy()
de_cp.update(cp_parameters)

arg_dict=dict(FA_OBL=fa_obl, DE_OBL=de_obl, FA_CP=fa_cp, DE_CP=de_cp)

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
        args=arg_dict[a_name].copy()
        args['n']=j
        print (i,j,f[0],a[0],args)
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

f=open("exercise_2.2_raw_data.pickle",'wb')
pickle.dump(results,f)
f.close()
#'''

## Outputting the results: ##
f=open("exercise_2.2_raw_data.pickle",'rb')
results=pickle.load(f)
f.close()

def sf(n,d=5):
  if(type(n)==int):
    r=str(n)
  else:
    #nr=round(n,d)
    #r=('%.'+str(d)+'g') % n
    r=('{: .'+str(d)+'f}').format(n)
    if(len(r)>(2+d)):
      new_size=max( 2+d , len(r.split('.')[0]) )
      r=r[:new_size]
  return r

tables=sorted(list(set([(i[2],i[3]) for i in results])),reverse=True)
for t in range(len(tables)):
  #print('Table '+str(t+1)+'. Convergency of the '+tables[t][1]+' algorithm applied to the '+tables[t][0]+' function.')
  print(r'\begin{table*}[h]')
  print(r' \centering')
  print(r' \caption{Algoritmo \textit{\textbf{'+tables[t][1].replace('_','\_')+r'}} aplicado à função \textit{'+tables[t][0].replace('_','\_')+r'}. Valores da função custo após 1000 iterações e '+str(number_of_repetitions)+' repetições.}')
  print(r' \begin{tabular}{llccccr}')
  print(r'  \toprule')
  print(r'   & & média & mediana & mínimo & desvio padrão & taxa de sucesso (\%) \\')
  print(r'  \midrule')
  if(tables[t][0]=='Rosenbrock'):
    tolerance=1e-1
  else:
    tolerance=1e-2
  for s in swarm_sizes:
    #print('S='+str(s),end='')
    for d in dimensions:
      costs=np.array([[i[5],i[6]] for i in results if (i[0],i[1],i[2],i[3])==(d,s,tables[t][0],tables[t][1])])
      mean_cost=np.mean(costs[:,0])
      median_cost=np.median(costs[:,0])
      min_cost=np.min(costs[:,0])
      stddev_cost=np.std(costs[:,0])
      success_rate=np.sum(np.abs(costs[:,0]-costs[:,1])<tolerance)/number_of_repetitions
      print('  S='+sf(s)+' & '+'N='+sf(d)+' & '+sf(mean_cost)+' & '+sf(median_cost)+' & '+sf(min_cost)+' & '+sf(stddev_cost)+' & '+sf(success_rate*100,2)+ r' \\')

  print(r'  \bottomrule')
  print(r' \end{tabular}')
  print(r'\end{table*}')

  print(r'\begin{table*}[h]')
  print(r' \centering')
  print(r' \caption{Algoritmo \textit{\textbf{'+tables[t][1].replace('_','\_')+r'}} aplicado à função \textit{'+tables[t][0].replace('_','\_')+r'}. Comparação das melhores soluções encontradas com o mínimo teórico utilizando 1000 iterações e '+str(number_of_repetitions)+' repetições.}')
  print(r' \begin{tabular}{llll}')
  print(r'  \toprule')
  print(r'  tipo & dimensoes & custo & solução correspondente\\')
  print(r'  \midrule')
  for d in dimensions:
    costs=np.array([[i[5],i[6]] for i in results if (i[0],i[2],i[3])==(d,tables[t][0],tables[t][1])])
    solutions=np.array([[i[8],i[9]] for i in results if (i[0],i[2],i[3])==(d,tables[t][0],tables[t][1])])
    bestindex=np.argmin(costs[:,0])
    besty=costs[bestindex,0]
    bestx=solutions[bestindex,0]
    print(  r'calculado & '+sf(d)+' & '+sf(besty)+r' & [ '+' '.join(map(sf,bestx))+r' ] \\')
    print(  r'teórico & '+sf(d)+' & '+sf(costs[bestindex,1])+r' & [ '+' '.join(map(sf,solutions[bestindex,1]))+r' ] \\')
    #print("Best solution found for N="+sf(d)+':',besty,"at point",bestx)
    #print("  Theoretical best:",costs[bestindex,1],"at point",solutions[bestindex,1])
  print(r'  \bottomrule')
  print(r' \end{tabular}')
  print(r'\end{table*}')