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
  """
  ## These variables are commented out on purpose to generate errors
  ##  in case they are used before being initialized.
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
  """

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
      x_history.append(s._bestx)
      y_history.append(s._besty)
      #protect against s.iterate_one incrementing the s._iter counter:
      if(s._iter==i):
        s._iter+=1
    return (y_history,x_history)


class PSO(Optimizer):
  name='PSO'
  description='Particle Swarm Optimization algorithm.'
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


class ABC(Optimizer):
  name='ABC'
  description='Artificial Bee Colony algorithm.'

  nb=24                 #Number of bees (employed bees plus onlooker bees).
  nf=12                 #Number of food sources. Default is nb/2.
  abandon_threshold=20  #Number of consecutive improvement trials that a food source undergoes before being abandoned.


  def __init__(self,costfunction,dimensions,lb,ub,maxiter=500,target_cost=None,
           nb=24,nf=12,abandon_threshold=20):
    #TODO: do input checking here.
    s=self

    #problem parameters:
    s.costfunction=costfunction
    s.dimensions=dimensions
    s.lb=lb.copy()
    s.ub=ub.copy()
    s.maxiter=maxiter
    s.target_cost=target_cost

    s.nb=nb                               # Number of bees (employed bees plus onlooker bees).
    s.nf=nf                               # Number of food sources. Default is nb/2.
    s.abandon_threshold=abandon_threshold # Number of consecutive improvement trials that a food source undergoes before being abandoned.

    #initial conditions:
    s._X = np.random.random((nf,dimensions))*(ub-lb)+lb # current food sources
    s._Y = s.costfunction(s._X)                         # current source cost
    s._trials = np.zeros(nf)                            # number of attempts to improve each solution

    s._bestidx=np.argmin(s._Y)       # index of best particle (food source) - watch out!!! if scout bees destroy this solution, the index is no longer valid, even though the besty and bestx variables are.
    s._bestx=s._X[s._bestidx].copy() # solution of best particle (food source) ever
    s._besty=s._Y[s._bestidx]        # cost of best particle (food source)

    s._iter=0
    return

  def iterate_one(self):
    #TODO: cleanup this code. refactor. document. use references and terminology from the article
    s=self

    #### Employed bee phase ####
    # The parameter to be changed is determined randomly
    parameters_to_change=np.random.randint(0,s.dimensions,s.nf)
    # A randomly chosen solution is used in producing a mutant solution of the solution i
    aux=np.arange(s.nf,dtype='int')
    neighbours=(np.random.randint(1,s.nf,s.nf)+aux)%s.nf # neighbour indices
    new_foods=s._X.copy()
    new_foods[(aux,parameters_to_change)]=s._X[(aux,parameters_to_change)]+(np.random.rand(s.nf)*2-1)*(s._X[(aux,parameters_to_change)]-s._X[(neighbours,parameters_to_change)])
    del aux,parameters_to_change,neighbours
    # if generated parameter value is out of boundaries, it is shifted onto the boundaries
    new_foods=np.maximum(new_foods,s.lb)
    new_foods=np.minimum(new_foods,s.ub)
    #evaluate new solution
    new_food_costs=s.costfunction(new_foods)
    #a greedy selection is applied between the current solution i and its mutant
    aux=(new_food_costs<s._Y) #mask to show where the cost has improved
    #If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i
    s._X=aux.reshape((-1,1))*new_foods+(1-aux.reshape((-1,1)))*s._X
    s._Y=aux*new_food_costs+(1-aux)*s._Y
    #increase trial count of solutions which not improve, zero the counter of the improved ones.
    s._trials=(1-aux)*(s._trials+1)
    del aux,new_food_costs,new_foods

    #### Calculate probabilities ####
    # A food source is chosen with the probability which is proportional to its quality
    #Different schemes can be used to calculate the probability values
    #For example prob(i)=fitness(i)/sum(fitness)
    #or in a way used in the metot below prob(i)=a*fitness(i)/max(fitness)+b
    #probability values are calculated by using fitness values and normalized by dividing maximum fitness value
    prob=(0.9*s._Y/np.max(s._Y))+0.1 #the higher the cost, the higher the probability for change

    #### Onlooker bee phase ####
    #TODO: fix the number of onlooker bees. It is currently hardcoded to be the same number of food sources.
    aux=np.where(prob>np.random.rand(s.nf))[0] #chosen food sources by probability
    n_aux=len(aux) #number of randomly chosen food sources
    # The parameter to be changed is determined randomly
    parameters_to_change=np.random.randint(0,s.dimensions,n_aux)
    # A randomly chosen solution is used in producing a mutant solution of the solution i
    # Randomly selected solution must be different from the solution i
    neighbours=(np.random.randint(1,s.nf,n_aux)+aux)%s.nf
    new_foods=s._X.copy()
    new_foods[(aux,parameters_to_change)]=s._X[(aux,parameters_to_change)]+(np.random.rand(n_aux)*2-1)*(s._X[(aux,parameters_to_change)]-s._X[(neighbours,parameters_to_change)])
    del aux,n_aux,parameters_to_change,neighbours
    # if generated parameter value is out of boundaries, it is shifted onto the boundaries
    new_foods=np.maximum(new_foods,s.lb)
    new_foods=np.minimum(new_foods,s.ub)
    #evaluate new solution
    new_food_costs=s.costfunction(new_foods)
    #a greedy selection is applied between the current solution i and its mutant
    aux=(new_food_costs<s._Y) #mask to show where the cost has improved
    #If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i
    s._X=aux.reshape((-1,1))*new_foods+(1-aux.reshape((-1,1)))*s._X
    s._Y=aux*new_food_costs+(1-aux)*s._Y
    #increase trial count of solutions which not improve, zero the counter of the improved ones.
    s._trials=(1-aux)*(s._trials+1)
    del aux,new_food_costs,new_foods

    #The best food source is identified
    s._bestidx=np.argmin(s._Y)
    if(s._Y[s._bestidx]<s._besty):
      #record only the best food source ever, even though there may be no more bees on it
      # this may depart from the original algorithm.
      #TODO: check the paper!
      s._bestx=s._X[s._bestidx].copy()
      s._besty=s._Y[s._bestidx]

    #### Scout bee phase ####
    #determine the food sources whose trial counter exceeds the "abandon_threshold" value.
    #In Basic ABC, only one scout is allowed to occur in each cycle
    aux=np.argmax(s._trials)
    if(s._trials[aux]>s.abandon_threshold):
      new_food=np.random.rand(s.dimensions)*(s.ub-s.lb)+s.lb
      new_cost=s.costfunction(np.array([new_food]))[0]
      #Since the best solution cannot be improved upon, it will eventually
      # hit the maximum trial count and be abandoned.
      # in this algorithm, scout bees destroy the best solution.
      s._X[aux]=new_food.copy()
      s._Y[aux]=new_cost
      s._trials[aux]=0
      #best food source is verified again, only against the scout:
      if(new_cost<s._besty):
        s._bestidx=aux
        s._bestx=s._X[s._bestidx].copy()
        s._besty=s._Y[s._bestidx]
    del aux

    #end of algorithm
    return


all_algorithms={i[0]:i[1] for i in vars().copy().items() if
                hasattr(i[1],'iterate_one') and
                hasattr(i[1],'run') and
                hasattr(i[1],'run_with_history')}

def test(algorithm,Fitnessfunc,dimensions,tolerance=1e-3,**kwargs):
  #TODO: check the fitnessfunction test and see if there are any tests that can be applied here
  #TODO: do a lot more tests
  #TODO: check if the name attribute is the same as the name of the class.
  f=Fitnessfunc.evaluate
  lb,ub=Fitnessfunc.default_bounds(dimensions)
  ymin,xmin=Fitnessfunc.default_minimum(dimensions)
  a=algorithm(f,dimensions,lb,ub,**kwargs)
  y,x=a.run_with_history()
  cost_delta=((y[-1]-ymin)**2).sum()**0.5
  solution_delta=((x[-1]-xmin)**2).sum()**0.5
  print('cost difference to ideal:     ',cost_delta)
  print('solution distance to ideal: ',solution_delta)
  print('converged within tolerance?   ',cost_delta<tolerance)
  print('Solution found:\n',x[-1])
  print('Theoretical best solution possible:\n',xmin)
  print('cost achieved:\n',y[-1])
  print('Theoretical best cost:\n',ymin)

#c=fitnessfunctions.Rosenbrock
#ndim=6
#test(PSO,c,ndim,maxiter=1000,tolerance=1e-2,n=30)
