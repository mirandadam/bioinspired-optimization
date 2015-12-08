#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np
import mode
import sys
sys.path.append('./ZDT')
sys.path.append('./DTLZ')

#import ZDT1 as fun
import DTLZ5_3obj as fun
ndim=20
lb=np.zeros(ndim)
ub=np.ones(ndim)
m=mode.MODE(fun.fit, n_dimensions=ndim, n_objectives=fun.number_of_objectives, lb=lb, ub=ub, maxiter=500,
            population_size=80, scaling_factor=0.5, crossover_probability=0.5, mutation_probability=1)


import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

if(fun.number_of_objectives==2):
  ax = fig.add_subplot(111)
  ax.plot(fun.pareto_front[:,0],fun.pareto_front[:,1])
  scatterplot=ax.scatter(m._Y[:,0],m._Y[:,1])
  def animate(i):
    print(i)
    m.iterate_one()
    a,b=np.max(m._Y,axis=0)
    ax.axis([0, a, 0, b])
    scatterplot.set_offsets(m._Y.copy())  
elif(fun.number_of_objectives==3):
  ax = fig.add_subplot(111, projection='3d')
  #ax.plot_trisurf(fun.pareto_front[::8,0],fun.pareto_front[::8,1],fun.pareto_front[::8,2])
  ax.scatter(fun.pareto_front[::8,0],fun.pareto_front[::8,1],fun.pareto_front[::8,2])
  scatterplot=ax.scatter(m._Y[:,0],m._Y[:,1],m._Y[:,2])
  def animate(i):
    print(i)
    m.iterate_one()
    a,b,c=np.max(m._Y,axis=0)
    ax.set_xlim3d([0, a])
    ax.set_ylim3d([0, b])
    ax.set_zlim3d([0, c])
    scatterplot._offsets3d=m._Y.transpose()

anim = animation.FuncAnimation(fig, animate, #init_func=init,
                               frames=500, interval=10, blit=False)

plt.show()
#'''
