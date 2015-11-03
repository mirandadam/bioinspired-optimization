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
import fitnessfunctions #needed for testing.

class Optimizer:
  '''
  #problem variables:
  costfunction=None      #the cost function to evaluate
  dimensions=None        #number of dimensions
  maxiter=500            #maximum number of iterations
  target_cost=None       #stopping criterion for cost
  lb=None                #range domain for fitness function - lower bound for each dimension
  ub=None                #range domain for fitness function - upper bound for each dimension
  
  #state variables:
  _X=None       #current particle solutions
  _Y=None       #current particle cost
  _bestidx=None #index of best particle in X
  _bestx=None   #solution of best particle in X
  _besty=None   #cost of best particle in X
  _iter=None    #current iteration
  '''
  
  def run(self):
    """
      Iterate the algorithm until a stop condition is met.
      Returns the final cost and the final solution found.
    """
    s=self
    while(s._iter<s.maxiter and
          (s.target_cost is None or s._ybest > s.target_cost) ):
      i=s._iter
      s.iterate_one()
      #protect against s.iterate_one incrementing the s._iter counter:
      if(s._iter==i):
        s._iter+=1
    return (s._besty,s._bestx)
    
  def run_with_history(self):
    """
      Iterate the algorithm until a stop condition is met.
      
      Returns the cost history and the solution history of each iteration in
       chronological order
    """
    s=self
    x_history=[]
    y_history=[]
    while(s._iter<s.maxiter and
          (s.target_cost is None or s._ybest > s.target_cost) ):
      i=s._iter
      s.iterate_one()
      x_history.append(s._xbest)
      y_history.append(s._ybest)
      #protect against s.iterate_one incrementing the s._iter counter:
      if(s._iter==i):
        s._iter+=1
    return (s._ybest,s._xbest)

class PSO(Optimizer):
  name='PSO'
  #algorithm tuning:
  n=10                   #number of particles
  w0=0.9                 #initial inertia coefficient (weight)
  wf=0.1                 #final inertia coefficient (weight)
  c1=2                   #cognitive coefficient
  c2=2                   #social coefficient
  max_v=5                #maximum velocity
  ini_v=max_v/10 #max_v/10 #initial velocity
  
  #state variables:
  _iter=0

  def __init__(self,costfunction,dimensions,lb,ub,maxiter=500,target_cost=None,
           n=10,w0=0.9,wf=0.1,c1=2,c2=2,max_v=5,ini_v=5/10):
    """
    The cost function has to take arrays with (m,n) shape as inputs, where
     m is the number of particles and n is the number of dimensions.
    lb is the lower bound for each dimension
    ub is the upper bound for each dimension
    """
    s=self
    #TODO: do some serious input checking here.
    
    #problem parameters:
    s.costfunction=costfunction
    s.dimensions=dimensions
    s.lb=lb.copy()
    s.ub=ub.copy()
    s.maxiter=maxiter
    s.target_cost=target_cost
    
    #algorithm tuning:
    s.n=n
    s.w0=w0
    s.wf=wf
    s.c1=c1
    s.c2=c2
    s.max_v=max_v
    s.ini_v=ini_v
    
    #initial conditions:
    s._X = np.random.random((n,dimensions))*(ub-lb)+lb # current particle solutions
    s._Y = s.costfunction(s._X)                  # current particle cost
    s._V = np.ones((n,dimensions))*ini_v               # current particle speeds
    
    s._Xmemory=s._X.copy()              # memory of best individual solution
    s._Ymemory=s._Y.copy()              # memory of best individual fitness
    s._bestidx=np.argmin(s._Ymemory)         # index of best particle in Xmemory
    s._bestx=s._Xmemory[s._bestidx].copy() # solution of best particle in Xmemory
    s._besty=s._Ymemory[s._bestidx]        # cost of best particle in Xmemory
    
    s._iter=0

  def iterate_one(self):
    s=self
    #calculating inertia weight:
    w=s.w0+s._iter*(s.wf-s.w0)/s.maxiter

    #particle movement:
    r1=np.random.random((s.n,s.dimensions))
    r2=np.random.random((s.n,s.dimensions))
    s._V= w*s._V + s.c1*r1*(s._Xmemory-s._X) + s.c2*r2*(s._bestx-s._X)

    #applying speed limit:
    vnorm=((s._V**2).sum(axis=1))**0.5 #norm of the speed
    aux=np.where(vnorm>s.max_v) #particles with velocity greater than expected
    s._V[aux]=s._V[aux]*s.max_v/(vnorm[aux].reshape((-1,1))) #clipping the speed to the maximum speed

    #update solutions:
    s._X=s._X+s._V

    #fitness value calculation:
    s._Y = s.costfunction(s._X)  # current particle cost
    
    #update memories:
    aux=np.where(s._Y<s._Ymemory)
    s._Xmemory[aux]=s._X[aux].copy()           # memory of best individual solution
    s._Ymemory[aux]=s._Y[aux].copy()           # memory of best individual fitness
    s._bestidx=np.argmin(s._Ymemory)           # index of best particle in Xmemory
    s._bestx=s._Xmemory[s._bestidx].copy()  # solution of best particle in Xmemory
    s._besty=s._Ymemory[s._bestidx]         # cost of best particle in Xmemory
    return

all_algorithms={i[0]:i[1] for i in vars().copy().items() if 
                hasattr(i[1],'iterate_one') and
                hasattr(i[1],'run') and
                hasattr(i[1],'run_with_history')}

def test(algo,Fitnessfunc,dimensions,tolerance=1e-3,**kwargs):
  #TODO: check the fitnessfunction test and see if there are any tests that can be applied here
  #TODO: do a lot more tests
  #TODO: check if the name attribute is the same as the name of the class.
  f=Fitnessfunc.evaluate
  lb,ub=Fitnessfunc.default_bounds(dimensions)
  ymin,xmin=Fitnessfunc.default_minimum(dimensions)
  a=algo(f,dimensions,lb,ub,**kwargs)
  y,x=a.run()
  cost_delta=((y-ymin)**2).sum()**0.5
  solution_delta=((x-xmin)**2).sum()**0.5
  print('cost difference to ideal:     ',cost_delta)
  print('solution difference to ideal: ',solution_delta)
  print('converged within tolerance?   ',cost_delta<tolerance)
  print('Solution found:\n',x)
  print('Theoretical best solution possible:\n',xmin)
  print('cost achieved:\n',y)
  print('Theoretical best cost:\n',ymin)
  
c=fitnessfunctions.Sphere
ndim=20
test(PSO,c,ndim,maxiter=1000)
