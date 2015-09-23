#!/usr/bin/python3
# -*- coding: utf8 -*-

#code licenced under the GPL

import numpy as np
import fitnessfunctions

#default Particle Swarm Optimization (PSO) parameters:
default_parameters={
  'S':10,                #number of particles
  'N':6,                 #number of dimensions
  'maxiter':100, #1000   #maximum number of iterations
  'w0':0.9,              #initial weight
  'wf':0.1,              #final weight
  'c1':2,                #cognitive coefficient
  'c2':2,                #social coefficient
  'max_v':5,             #maximum velocity
  'ini_v':5/10,#max_v/10 #initial velocity
  'x_max':8,             #range domain for fitness function - inferior limit (assumes the same for all dimensions)
  'x_min':-8,            #range domain for fitness function - superior limit (assumes the same for all dimensions)
  'threshold':1e-10
}


#np.random.seed(20150904)

def pso(f,parameters):
  global default_parameters
  p=default_parameters.copy()
  p.update(parameters)
  S=p['S']
  N=p['N']
  maxiter=p['maxiter']
  w0=p['w0']
  wf=p['wf']
  c1=p['c1']
  c2=p['c2']
  max_v=p['max_v']
  ini_v=p['ini_v']
  x_max=p['x_max']
  x_min=p['x_min']
  threshold=p['threshold']

  x=np.random.random((S,N))*(x_max-x_min)+x_min
  y=np.ones((S,N),dtype='float64')*1e10
  v=np.ones((S,N),dtype='float64')*ini_v
  f_ind=np.ones(S,dtype='float64')*1e10 # initialize best local fitness 
  bestfitness=[]

  w= w0 #current weight
  slope = (wf-w0)/maxiter #increment that will take w from w0 to wf in the specified number of iteractions
  k = 1
  while k<=maxiter:
    #fitness value calculation:
    fx=np.array([f(i) for i in x])
    
    #memory update:
    aux=np.where(fx<f_ind)
    f_ind[aux]=fx[aux]
    y[aux]=x[aux]
    best=np.min(f_ind)
    aux=np.where(f_ind==best)[0][0]
    bestfitness.append(best)
    ys=y[aux]

    #particle movement:  
    r1=np.random.random((S,N))
    r2=np.random.random((S,N))
    v=w*v+c1*r1*(y-x)+c2*r2*(np.array([ys - i for i in x]))
    #speed limit:
    v2=(v*v).sum(axis=1)
    aux=np.where(v2>(max_v*max_v))
    v[aux]=v[aux]/(v2[aux]/max_v).reshape((-1,1))
    #position update:
    x=x+v

    #inertia update:
    w=w+slope
    #counter update:
    k=k+1

  print(ys)
  print(best)
  return(bestfitness)

bestfitness=pso(fitnessfunctions.ackley,{'x_min':-30,'x_max':30,'N':12,'S':30,'maxiter':1000 })

import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


fig = plt.figure()
ax=fig.add_subplot(111)
ax.plot(np.log10(bestfitness))
plt.show()
#ax2=fig.add_subplot(122)


