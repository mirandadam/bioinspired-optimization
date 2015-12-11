#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np

import base

class MODE:

  def __init__(self,cost_function,n_dimensions,n_objectives,lb,ub,maxiter=500,
               population_size=80,scaling_factor=0.5,crossover_probability=0.2,mutation_probability=1.0):
    s=self
    s.cost_function=cost_function
    s.n_dimensions=n_dimensions #number of decision variables
    s.n_objectives=n_objectives #number of objectives
    s.lb=lb.copy() #lower search space bounds
    s.ub=ub.copy() #uppser search space bounds
    s.maxiter=maxiter
    s.population_size=population_size
    s.scaling_factor=scaling_factor #Scaling factor of the DE algorithm
    s.crossover_probability=crossover_probability #crossover probability of the DE algorithm
    s.mutation_probability=mutation_probability #this is mentioned in the article (TODO: cite), but was not present in the DE code studied so far

    #initial conditions:
    s._X = np.random.random((population_size,n_dimensions))*(ub-lb)+lb # current individuals
    s._Y = s.cost_function(s._X)                        # current cost
    assert(s._Y.shape[1]==n_objectives)
    s._iter=0

  def _mutate_and_crossover(self):
    """
      Selects neighbours and apply mutation.
    """

    s=self
    n=s.population_size
    assert(n>3) #only works with 4 or more individuals

    #getting random neighbours to permutate
    neighbours=np.array([np.random.permutation(n-1)[:3] for i in range(n)]).transpose()
    neighbours=(neighbours+np.arange(n)+1)%n
    n0=np.arange(n)    #self
    n1=neighbours[0,:] #neighbours different from self
    n2=neighbours[1,:] #neighbours different from self and n1
    n3=neighbours[2,:] #neighbours different from self, n1 and n2
    '''
    #DEBUG:
    assert((n0!=n1).all())
    assert((n0!=n2).all())
    assert((n0!=n3).all())
    assert((n1!=n2).all())
    assert((n1!=n3).all())
    assert((n2!=n3).all())
    '''
    #mutation:
    aux=np.random.rand(s.population_size) #probability calculation for mutation
    aux= (aux<s.mutation_probability).reshape(-1,1) #individuals to mutate to crossover
    m=aux*(s._X[n1]+s.scaling_factor*(s._X[n2]-s._X[n3])) + (1-aux)*s._X
    

    #bound checking:
    m=np.minimum(s.ub,m)
    m=np.maximum(s.lb,m)

    aux=np.random.rand(s.population_size,s.n_dimensions) #probability calculation for crossover
    aux= (aux<s.crossover_probability) #dimensions to crossover
    crossover_X=aux*m+(1-aux)*s._X  #take dimension values from the mutated population or from the old population
    return crossover_X


  def iterate_one(self):
    s=self

    parents  = s._X.copy()
    children = s._mutate_and_crossover()
    new_population=np.concatenate((parents,children),axis=0)
    new_Y=s.cost_function(new_population)

    truncated_Y,rank,cf,indices=base.truncate(new_Y,s.population_size)

    s._X=new_population[indices].copy()
    s._Y=truncated_Y.copy()

    return

  def run(self):
    """
      Iterate the algorithm until a stop condition is met.
      Returns the final cost and the final solution found.
    """
    s=self
    while(s._iter<s.maxiter):
      i=s._iter
      s.iterate_one()
      #protect against s.iterate_one incrementing the s._iter counter:
      if(s._iter==i):
        s._iter+=1
    return (s._Y,s._X)
