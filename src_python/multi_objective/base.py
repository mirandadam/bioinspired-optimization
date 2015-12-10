#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np

def nonDominatedRankV1(Y):
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

def nonDominatedRankV2(Y):#same as v1 but much faster
  population_size=len(Y)
  rank=np.zeros(population_size,'int')
  number_of_classes=0
  domination_matrix=np.zeros((population_size,population_size),dtype='bool')
  #calculating domination matrix:
  for i in range(population_size):
    domination_matrix[i,:]=(Y[i]<Y).any(axis=1)*(Y[i]<=Y).all(axis=1)
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
    rank=nonDominatedRankV2(Y)
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
  rank=nonDominatedRankV2(Y)
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

def component_sort_order(m):
  #the resulting array contains integers representing the sort order of the value of each dimension in relation to all the elements
  #if x is:
  #array([[ 0.09260602,  0.50714411,  0.28942794,  0.11886156,  0.30032682],
  #       [ 0.71064396,  0.62893067,  0.85888687,  0.48403725,  0.17259805],
  #       [ 0.95584438,  0.39083918,  0.33461762,  0.51868568,  0.32926818]])
  #then component_sort_order(x) is:
  #array([[0, 1, 0, 0, 1],
  #       [1, 2, 2, 1, 0],
  #       [2, 0, 1, 2, 2]])
  count,dims=m.shape
  r=np.zeros((count,dims),'int')
  for i in range(dims):
    r[np.argsort(m[:,i]),i]=np.arange(count)
  return r