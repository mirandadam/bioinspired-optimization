#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np

number_of_objectives=2

def fit(var):
  assert(len(var.shape)==2) #assert that the input has two dimensions
  population_size,n=var.shape #population size, number of dimensions
  g=1+9*np.sum(var[:,1:],axis=1)/(n-1)
  r=np.zeros((population_size,2))
  x0=var[:,0]
  r[:,0]=x0
  r[:,1]=g*(1-(x0/g)**0.5)
  return r

fname=__file__.rsplit('.py',1)[0]+'.txt'
print('loading',fname)
pareto_front=np.loadtxt(fname,delimiter=',')

#print(fit(np.zeros((3,10))))
#from matplotlib import pyplot
#pyplot.plot(pareto_front[:,0],pareto_front[:,1])
