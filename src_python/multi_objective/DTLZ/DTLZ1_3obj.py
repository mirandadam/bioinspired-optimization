#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np

number_of_objectives=3

def fit(var):
  assert(len(var.shape)==2) #assert that the input has two dimensions
  population_size,n=var.shape #population size, number of dimensions
  k=5
  assert(n>=k)
  g=100*(k + np.sum((var[:,n-k:]-0.5)**2 - np.cos(20*np.pi*(var[:,n-k:]-0.5)),axis=1) )
  r=np.zeros((population_size,3))
  r[:,0]= (1+g) * 0.5 * var[:,0] * var[:,1]
  r[:,1]= (1+g) * 0.5 * var[:,0] * (1-var[:,1])
  r[:,2]= (1+g) * 0.5 * (1-var[:,0])
  return r


fname=__file__.rsplit('.py',1)[0]+'.txt'
print('loading',fname)
pareto_front=np.loadtxt(fname,delimiter='\t',usecols=(0,1,2))

#print(fit(np.zeros((3,10))))
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_trisurf(pareto_front[::12,0], pareto_front[::12,1], pareto_front[::12,2])
#plt.show()