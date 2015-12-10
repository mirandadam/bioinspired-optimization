#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np

def nonDominatedRank(Y):
  population_size=len(Y)
  rank=np.zeros(population_size,'int')
  number_of_classes=0
  domination_matrix=np.zeros((population_size,population_size),dtype='bool')
  #calculating domination matrix:
  for i in range(population_size):
    for j in range(population_size):
      domination_matrix[i,j]=(Y[i]<Y[j]).any()*(Y[i]<=Y[j]).all()
      #domination_matrix[i,j] will answer the question: does element i dominate element j?
  non_dominated=1-np.any(domination_matrix,axis=0)
  aux=np.where(non_dominated)[0]
  while np.any(non_dominated):
    number_of_classes+=1
    rank[aux]=number_of_classes
    for i in aux:
      domination_matrix[i,:]=0 #update these to dominate no other
      domination_matrix[i,i]=1 #update the element to be dominated by itself
    non_dominated=1-np.any(domination_matrix,axis=0)
    aux=np.where(non_dominated)[0]
  return rank

def computecf(Y,rank=None):
  population_size, number_of_dimensions = Y.shape
  cf=np.zeros(population_size)
  if rank is None:
    rank=nonDominatedRank(Y)
  for r in range(1,rank.max()+1):
    indices=np.where(rank==r)[0]
    for d in range(number_of_dimensions):
      sorted_indices=indices[np.argsort(Y[indices,d])]
      cf[sorted_indices[0]]=np.inf #the crowding distance of the extremes is infinity
      cf[sorted_indices[-1]]=np.inf #the crowding distance of the extremes is infinity
      dimension_range=Y[sorted_indices[-1],d]-Y[sorted_indices[0],d]
      dimension_range=max(1e-6,dimension_range)#avoid dividing by zero in case there is no variance in the dimension
      cf[sorted_indices[1:-1]]+=(Y[sorted_indices[2:],d]-Y[sorted_indices[:-2],d])/dimension_range
  return cf.copy()

def truncate(Y,N):
  #steps:
  #1.order solutions in Y by rank
  #2.within each rank, order solutions by descending crowding factor (cf)
  #3.truncate the resulting solution vector to exactly N elements

  #compute rank:
  rank=nonDominatedRank(Y)
  #print(np.bincount(rank)) #this line is for debugging. It shows how many individuals are in each rank.
  #compute crowding factor:
  cf=computecf(Y,rank)

  #sort rank, and within each rank sort by descending crowding factor (cf)
  aux=np.lexsort((-cf,rank))
  # the syntax of the lexsort command is confusing. If in doubt, please make
  # the following test:
  #  aux=np.lexsort((-cf,rank))
  #  p=np.array([rank,cf]).transpose()
  #  print(p[aux]) #correct result
  # then try:
  #  aux=np.lexsort((rank,-cf)) #switched parameters
  #  p=np.array([rank,cf]).transpose()
  #  print(p[aux]) #incorrect result

  #truncate the sorted indices to contain N elements:
  aux=aux[:N]
  #return the N "best" solutions in Y according to rank and cf.
  #also return the rank of each solution and the corresponding crowding factor
  return [Y[aux].copy(),rank[aux].copy(),cf[aux].copy(),aux.copy()]

class MODE:
  #function OUT=MODE(MODEDat)
  # Reading parameters from MODEDat
  #Generaciones  = MODEDat.MAXGEN;    % Maximum number of generations.
  #Xpop          = MODEDat.XPOP;      % Population size.
  #Nvar          = MODEDat.NVAR;      % Number of decision variables.
  #Nobj          = MODEDat.NOBJ;      % Number of objectives.
  #Bounds        = MODEDat.FieldD;    % Optimization bounds.
  #Initial       = MODEDat.Initial;   % Initialization bounds.
  #ScalingFactor = MODEDat.Esc;       % Scaling fator in DE algorithm.
  #CrossOverP    = MODEDat.Pm;        % Crossover probability in DE algorithm.
  #mop           = MODEDat.mop;       % Cost function.

  def __init__(self,cost_function,n_dimensions,n_objectives,lb,ub,maxiter=500,
               population_size=80,scaling_factor=0.5,crossover_probability=0.2,mutation_probability=1.0):
    s=self
    s.cost_function=cost_function
    s.n_dimensions=n_dimensions
    s.n_objectives=n_objectives
    s.lb=lb.copy()
    s.ub=ub.copy()
    s.maxiter=maxiter
    s.population_size=population_size
    s.scaling_factor=scaling_factor
    s.crossover_probability=crossover_probability
    s.mutation_probability=mutation_probability

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

    truncated_Y,rank,cf,indices=truncate(new_Y,s.population_size)

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

#inter generational distance performance - average distance to the pareto front:
def igd_performance(fitness_tuples,pareto_front):
  aux=0
  for i in fitness_tuples:
    aux+=np.min(((pareto_front-i)**2).sum(axis=1))**0.5
  return aux/len(fitness_tuples)

#measure of how spread the fitness functions are.
#this is the standard deviation of the euclidean distance of each fitness tuple to the nearest one
def spacing_performance(fitness_tuples):
  di=np.zeros(len(fitness_tuples))
  mx=np.max(fitness_tuples)-np.min(fitness_tuples)
  mx=mx*np.sqrt(fitness_tuples.shape[1]) #mx is the maximum possible distance in this set
  for i in range(len(fitness_tuples)):
    temp=np.sum((fitness_tuples-fitness_tuples[i])**2,axis=1)**0.5
    temp[i]=mx
    di[i]=np.min(temp)
  return np.std(di) #standard deviation
