#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np
import mode
import base
import sys
import time
sys.path.append('./ZDT')
sys.path.append('./DTLZ')

import ZDT1
import ZDT2
import ZDT3
import ZDT4
import DTLZ1_3obj
import DTLZ2_3obj
import DTLZ3_3obj
import DTLZ5_3obj

test_set=[
 {'name':'ZDT1' ,'fun':ZDT1, 'ndim':30, 'maxiter':500},
 {'name':'ZDT2' ,'fun':ZDT2, 'ndim':30, 'maxiter':500},
 {'name':'ZDT3' ,'fun':ZDT3, 'ndim':30, 'maxiter':500},
 {'name':'ZDT4' ,'fun':ZDT4, 'ndim':10, 'maxiter':500},
 {'name':'DTLZ1_3obj' ,'fun':DTLZ1_3obj, 'ndim':10, 'maxiter':300},
 {'name':'DTLZ2_3obj' ,'fun':DTLZ2_3obj, 'ndim':10, 'maxiter':300},
 {'name':'DTLZ3_3obj' ,'fun':DTLZ3_3obj, 'ndim':10, 'maxiter':300},
 {'name':'DTLZ5_3obj' ,'fun':DTLZ5_3obj, 'ndim':10, 'maxiter':300}
]

number_of_repetitions=8

samples=[]
for t in test_set:
  ndim=t['ndim']
  maxiter=t['maxiter']
  fit=t['fun'].fit
  number_of_objectives=t['fun'].number_of_objectives
  pareto_front=t['fun'].pareto_front
  lb=np.zeros(ndim)
  ub=np.ones(ndim)
  
  for n in range(number_of_repetitions):
    m=mode.MODE(fit, n_dimensions=ndim, n_objectives=number_of_objectives, lb=lb, ub=ub, maxiter=maxiter,
                population_size=80, scaling_factor=0.5, crossover_probability=0.5, mutation_probability=1)              
    for i in range(maxiter):
      m.iterate_one()
      #print(i)
      spacing= base.spacing_performance(m._Y)
      igd=     base.igd_performance(m._Y,pareto_front)
      s=[t['name'],n,i,spacing,igd]
      samples.append(s)
      print(s)


import pickle
f=open('samples_'+str(time.time())+'.pickle','wb')
pickle.dump(samples,f)
f.close()


'''
##### Running and plotting the results #####

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
Y=fun.fit(np.random.rand(10000,ndim))
if(fun.number_of_objectives==2):
  ax = fig.add_subplot(111)
  ax.scatter(Y[:,0],Y[:,1])
elif(fun.number_of_objectives==3):
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(Y[:,0],Y[:,1],Y[:,2])


fig = plt.figure()
if(fun.number_of_objectives==2):
  ax = fig.add_subplot(111)
  ax.plot(fun.pareto_front[:,0],fun.pareto_front[:,1])
  scatterplot=ax.scatter(m._Y[:,0],m._Y[:,1])
  def animate(i):
    #print(i)
    m.iterate_one()
    print(mode.igd_performance(m._Y,fun.pareto_front))
    a,b=np.max(m._Y,axis=0)
    ax.axis([0, a, 0, b])
    scatterplot.set_offsets(m._Y.copy())  
elif(fun.number_of_objectives==3):
  ax = fig.add_subplot(111, projection='3d')
  #ax.plot_trisurf(fun.pareto_front[::8,0],fun.pareto_front[::8,1],fun.pareto_front[::8,2])
  ax.scatter(fun.pareto_front[::8,0],fun.pareto_front[::8,1],fun.pareto_front[::8,2],c='g')
  scatterplot=ax.scatter(m._Y[:,0],m._Y[:,1],m._Y[:,2])
  def animate(i):
    #print(i)
    m.iterate_one()
    #print(mode.spacing_performance(m._Y))
    print(mode.igd_performance(m._Y,fun.pareto_front))
    a,b,c=np.max(m._Y,axis=0)
    ax.set_xlim3d([0, a])
    ax.set_ylim3d([0, b])
    ax.set_zlim3d([0, c])
    scatterplot._offsets3d=m._Y.transpose()

anim = animation.FuncAnimation(fig, animate, #init_func=init,
                               frames=500, interval=10, blit=False)

plt.show()
#'''
