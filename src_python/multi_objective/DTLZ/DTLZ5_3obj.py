#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np

number_of_objectives=3

def fit(var):
  assert(len(var.shape)==2) #assert that the input has two dimensions
  population_size,n=var.shape #population size, number of dimensions
  k=10
  g=np.sum((var[:,n-k:]-0.5)**2,axis=1).reshape(-1,1)
  t=var.copy()
  t[:,1:]=(1+2*g*var[:,1:])/(2*(1+g))
  r=np.zeros((population_size,3))
  g=g.ravel()
  r[:,0]= (1+g) * np.cos(t[:,0]*np.pi/2) * np.cos(t[:,1]*np.pi/2)
  r[:,1]= (1+g) * np.cos(t[:,0]*np.pi/2) * np.sin(t[:,1]*np.pi/2)
  r[:,2]= (1+g) * np.sin(t[:,0]*np.pi/2)
  return r

fname=__file__.rsplit('.py',1)[0]+'.txt'
print('loading',fname)
pareto_front=np.loadtxt(fname,delimiter=' ',usecols=(0,1,2))

#print(fit(np.zeros((5,10))))
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_trisurf(pareto_front[::12,0], pareto_front[::12,1], pareto_front[::12,2])
#plt.show()
