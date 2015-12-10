#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np
import mopso
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

#"""
test_set=[
 {'name':'MOPSO ZDT1' ,'fun':ZDT1, 'ndim':30, 'maxiter':500},
 {'name':'MOPSO ZDT2' ,'fun':ZDT2, 'ndim':30, 'maxiter':500},
 {'name':'MOPSO ZDT3' ,'fun':ZDT3, 'ndim':30, 'maxiter':500},
 {'name':'MOPSO ZDT4' ,'fun':ZDT4, 'ndim':10, 'maxiter':500},
 {'name':'MOPSO DTLZ1_3obj' ,'fun':DTLZ1_3obj, 'ndim':10, 'maxiter':300},
 {'name':'MOPSO DTLZ2_3obj' ,'fun':DTLZ2_3obj, 'ndim':10, 'maxiter':300},
 {'name':'MOPSO DTLZ3_3obj' ,'fun':DTLZ3_3obj, 'ndim':10, 'maxiter':300},
 {'name':'MOPSO DTLZ5_3obj' ,'fun':DTLZ5_3obj, 'ndim':10, 'maxiter':300}
]

number_of_repetitions=32

'''
#DEBUG:
number_of_repetitions=1
test_set=[
 {'name':'ZDT1' ,'fun':ZDT1, 'ndim':30, 'maxiter':500},
  ]
'''

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
    m=mopso.MOPSO(fit, n_dimensions=ndim, n_objectives=number_of_objectives, lb=lb, ub=ub, maxiter=maxiter,
                  population_size=80,
                  initial_inertia=0.9,
                  final_inertia=0.1,
                  social_coefficient=2.05,
                  cognitive_coefficient=0.0,
                  repulsion_coefficient=0.0,
                  maximum_velocity=0.3)
    for i in range(maxiter):
      m.iterate_one()
      m._iter+=1
      #print(i)
      spacing= base.spacing_performance(m._Y)
      igd=     base.igd_performance(m._Y,pareto_front)
      s=[t['name'],n,i,spacing,igd]
      samples.append(s)
      print(s)

import pickle
f=open('mopso_samples_'+str(time.time())+'.pickle','wb')
pickle.dump(samples,f)
f.close()
#"""


'''
##### Running and plotting the results #####
#fun=ZDT1
#fun=ZDT2
#fun=ZDT3
fun=ZDT4
#fun=DTLZ1_3obj
#fun=DTLZ2_3obj
#fun=DTLZ3_3obj
#fun=DTLZ5_3obj
fit=fun.fit
number_of_objectives=fun.number_of_objectives
ndim=10
lb=np.zeros(ndim)
ub=np.ones(ndim)
maxiter=300

m=mopso.MOPSO(fit, n_dimensions=ndim, n_objectives=number_of_objectives, lb=lb, ub=ub, maxiter=maxiter,
                  population_size=80,
                  initial_inertia=0.9,
                  final_inertia=0.1,
                  social_coefficient=2.05,
                  cognitive_coefficient=0.0,
                  repulsion_coefficient=0.0,
                  maximum_velocity=0.3)#,#ub-lb/10
                  #initial_velocity=0.01)

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
  scatterplot2=ax.scatter(m._global_bestY[:,0],m._global_bestY[:,1],c='r')
  #scatterplot3=ax.scatter(m._individual_bestY[:,0],m._individual_bestY[:,1],c='g')
  def animate(i):
    #print(i)
    m.iterate_one()
    m._iter+=1
    print(m._iter,base.igd_performance(m._Y,fun.pareto_front))
    a,b=np.max(m._Y,axis=0)
    c,d=np.max(m._global_bestY,axis=0)
    ax.axis([0, max(a,c), 0, max(b,d)])
    scatterplot.set_offsets(m._Y.copy())
    scatterplot2.set_offsets(m._global_bestY.copy())
    #scatterplot3.set_offsets(m._individual_bestY.copy())
elif(fun.number_of_objectives==3):
  ax = fig.add_subplot(111, projection='3d')
  #ax.plot_trisurf(fun.pareto_front[::8,0],fun.pareto_front[::8,1],fun.pareto_front[::8,2])
  aux=np.random.permutation(len(fun.pareto_front))[:1000]
  ax.scatter(fun.pareto_front[aux,0],fun.pareto_front[aux,1],fun.pareto_front[aux,2],c='g')
  #ax.scatter(fun.pareto_front[::8,0],fun.pareto_front[::8,1],fun.pareto_front[::8,2],c='g')
  scatterplot=ax.scatter(m._Y[:,0],m._Y[:,1],m._Y[:,2])
  scatterplot2=ax.scatter(m._global_bestY[:,0],m._global_bestY[:,1],m._global_bestY[:,2],c='r')
  def animate(i):
    #print(i)
    m.iterate_one()
    m._iter+=1
    #print(base.spacing_performance(m._Y))
    print(m._iter, base.igd_performance(m._Y,fun.pareto_front))
    a,b,c=np.max(m._Y,axis=0)
    d,e,f=np.max(m._global_bestY,axis=0)
    #ax.set_xlim3d([0, max(a,d)])
    #ax.set_ylim3d([0, max(b,e)])
    #ax.set_zlim3d([0, max(c,f)])
    ax.set_xlim3d([0, d])
    ax.set_ylim3d([0, e])
    ax.set_zlim3d([0, f])
    scatterplot._offsets3d=m._Y.transpose()
    scatterplot2._offsets3d=m._global_bestY.transpose()

anim = animation.FuncAnimation(fig, animate, #init_func=init,
                               frames=300, interval=100, blit=False)

plt.show()

m._Y
#'''
