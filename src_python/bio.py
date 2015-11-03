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
  #problem variables:
  costfunction=None      #the cost function to evaluate
  dimensions=None        #number of dimensions
  maxiter=500            #maximum number of iterations
  target_cost=None       #stopping criterion for cost
  lb=None                #range domain for fitness function - lower bound for each dimension
  um=None                #range domain for fitness function - upper bound for each dimension
  
  #state variables:
  _X=None       #current particle solutions
  _Y=None       #current particle cost
  _bestidx=None #index of best particle in X
  _xbest=None   #solution of best particle in X
  _ybest=None   #cost of best particle in X
  _iter=None    #current iteration
  
  def run(self):
    """
      Iterate the algorithm until a stop condition is met.
      Returns the final cost and the final solution found.
    """
    while(self._iter<self.maxiter and
          (self.target_cost is None or self._ybest > self.target_cost) ):
      self.iterate_one()
      self._iter+=1
    return (self._ybest,self._xbest)
    
  def run_with_history(self):
    """
      Iterate the algorithm until a stop condition is met.
      
      Returns the cost history and the solution history of each iteration in
       chronological order
    """

class PSO(Optimizer):
  #algorithm tuning:
  n=10                   #number of particles
  w0=0.9                 #initial inertia coefficient (weight)
  wf=0.1                 #final inertia coefficient (weight)
  c1=2                   #cognitive coefficient
  c2=2                   #social coefficient
  max_v=5                #maximum velocity
  ini_v=max_v/10 #max_v/10 #initial velocity

  def init(self,costfunction,dimensions,lb,ub,maxiter=500,target_cost=None,
           n=10,w0=0.9,wf=0.1,c1=2,c2=2,max_v=5,ini_v=5/10):
    """
    The cost function has to take arrays with (m,n) shape as inputs, where
     m is the number of particles and n is the number of dimensions.
    lb is the lower bound for each dimension
    ub is the upper bound for each dimension
    """
    #TODO: do some serious input checking here.
    
    #problem parameters:
    self.costfunction=costfunction
    self.dimensions=dimensions
    self.lb=lb.copy()
    self.ub=ub.copy()
    self.maxiter=maxiter
    self.target_cost=target_cost
    
    #algorithm tuning:
    self.n=n
    self.w0=w0
    self.wf=wf
    self.c1=c1
    self.c2=c2
    self.max_v=max_v
    self.ini_v=ini_v
    
    #initial conditions:
    self._X = np.random.random((n,dimensions))*(ub-lb)+lb # current particle solutions
    self._Y = self.costfunction(self._X)                  # current particle cost
    self._V = np.ones((n,dimensions))*ini_v               # current particle speeds
    
    self._Xmemory=self._X.copy()              # memory of best individual solution
    self._Ymemory=self._Y.copy()              # memory of best individual fitness
    self._bestidx=np.argmin(self.__Ymemory)         # index of best particle in Xmemory
    self._bestx=self._Xmemory[self._bestidx].copy() # solution of best particle in Xmemory
    self._besty=self._Ymemory[self._bestidx]        # cost of best particle in Xmemory
    
    self._iter=0

  def iterate_one(self):
    #calculating inertia weight:
    w=self.w0+self._iter*(self.wf-self.w0)/self.maxiter

    #particle movement:
    r1=np.random.random((self.n,self.dimensions))
    r2=np.random.random((self.n,self.dimensions))
    self._V= w*self._V + self.c1*r1*(self._Xmemory-self._X) + self.c2*r2*(self._bestx-self._X)

    #applying speed limit:
    vnorm=((self._V**2).sum(axis=1))**0.5 #norm of the speed
    aux=np.where(vnorm>self.max_v) #particles with velocity greater than expected
    self._V[aux]=self._V[aux]*max_v/vnorm[aux] #clipping the speed to the maximum speed

    #update solutions:
    self._X=self._X+self._V

    #fitness value calculation:
    self._Y = self.costfunction(self._X)  # current particle cost
    
    #update memories:
    aux=np.where(self._Y<self._Ymemory)
    self._Xmemory[aux]=self._X[aux].copy()           # memory of best individual solution
    self._Ymemory[aux]=self._Y[aux].copy()           # memory of best individual fitness
    self._bestidx=np.argmin(self._Ymemory)           # index of best particle in Xmemory
    self._bestx=self._Xmemory[self._bestidx].copy()  # solution of best particle in Xmemory
    self._besty=self._Ymemory[self._bestidx]         # cost of best particle in Xmemory
    return

