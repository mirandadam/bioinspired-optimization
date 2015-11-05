#!/usr/bin/python3
# -*- coding: utf8 -*-
"""
@author: Daniel Araujo Miranda
Licence: Free and open source under the GPL license
"""

import numpy as np
import fitnessfunctions

#differential evolution algorithm

#X = matriz 
#M = Tamaho da população
#F = Fator de mutação
#m = número do indivíduo atual

def mutate(X,F,direction=1):#div):
  #div_min = 0.25
  #div_max = 0.50

  #direction=1
  M=X.shape[0]
  assert(M>3) #only works with 4 or more individuals

  #if(div<div_min):
  #  direction=-1
  #elif(div>div_max):
  #  direction=1
  
  #getting random neighbours to permutate
  n1=(np.random.randint(0,M-1,(M,))+np.arange(M)+1)%M #neighbours different from self
  n2=(np.random.randint(0,M-1,(M,))+np.arange(M)+1)%M #neighbours different from self and n1
  aux=(n2==n1)
  w=np.where(aux)
  s=w[0].shape[0]
  while(s>0): #finding new neighbours if n2==n1
    n2[w]=((np.random.randint(0,M-1,(s,))+w[0]+1)%M)[:]
    aux=(n2==n1)
    s=aux.sum()
    w=np.where(aux)
  n3=(np.random.randint(0,M-1,(M,))+np.arange(M)+1)%M #neighbours different from self, n1 and n2
  aux=((n3==n2)+(n3==n1))
  w=np.where(aux)
  s=w[0].shape[0]
  while(s>0):
    n3[w]=((np.random.randint(0,M-1,(s,))+w[0]+1)%M)[:]
    aux=((n3==n2)+(n3==n1))
    w=np.where(aux)
    s=w[0].shape[0]
  
  return X[n1]+direction*F*(X[n2]-X[n3])
  
#default Differential Evolution parameters
default_parameters={
  #Algorithm tuning:
  'N':10,  #number of variables (dimensions) of the problem
  'M':50,  #population size
  'F':0.5, #mutation factor
  'C':0.9, #crossover rate
  'maxiter':500, #maximum number of iterations to run
  
  #Problem parameters:
  'ub':np.ones(10)*+8, #upper bound of the parameters
  'lb':np.ones(10)*-8 #lower bound of the parameters
}


def de(f=fitnessfunctions.rastrigin_m, parameters=default_parameters):
  global default_parameters
  p=default_parameters.copy()
  p.update(parameters)
  N=p['N'] 
  M=p['M'] 
  F=p['F']
  C=p['C']
  maxiter=p['maxiter']
  ub=p['ub']
  lb=p['lb']
  del p, parameters
  
  cost_history=[];

  aux_ub=np.array([ub]).repeat(M,axis=0)
  aux_lb=np.array([lb]).repeat(M,axis=0)
  X=aux_lb+np.random.rand(M,N)*(aux_ub-aux_lb)
  del aux_ub, aux_lb
  
  cost_history=[]
  #X[0,:]=420.9687 #schwefel minimum
  
  for i in range(maxiter):
    #Mutation:
    V=mutate(X,F,direction=1)
    #If generated parameter value is out of boundaries, it is shifted onto the boundaries:
    V=np.minimum(ub,V)
    V=np.maximum(lb,V)
    
    # Crossover put the result in the U matrix
    jrand=np.random.randint(N,size=(M,)) #dimensions chosen for each individual
    R1=np.random.rand(M,N) #calculating random variables
    aux=(R1<C) #dimensions to crossover based on probability
    aux[[np.arange(M),jrand]]=True #choose one dimension of each individual to crossover regardless os probability 
    U=aux*V+(aux==0)*X
    
    # Selection
    fx=f(X)
    fu=f(U)
    aux=(fu<fx) #individuals where the cost function decreased
    Y=np.minimum(fx,fu)
    X[aux]=U[aux]
    
    bestY=Y.min()
    cost_history.append(bestY)
  return cost_history


#quick test:
#cost_history=de(fitnessfunctions.rosenbrock_m,
#cost_history=de(fitnessfunctions.schwefel_m,
cost_history=de(fitnessfunctions.sphere_m,
                 {'N':10,  #number of variables (dimensions) of the problem
                  'M':50,  #population size
                  'F':0.5, #mutation factor
                  'C':0.9, #crossover rate
                  'maxiter':1000, #maximum number of iterations to run
                  #Problem parameters:
                  'ub':np.ones(10)*+8, #upper bound of the parameters
                  'lb':np.ones(10)*-8 })

import numpy
import matplotlib.pyplot as plt

fig = plt.figure()
ax1=fig.add_subplot(211)
#ax1.plot(np.log10(cost_history))
ax1.plot(cost_history)
ax1.set_ylabel("$mincost$")
ax2=fig.add_subplot(212)
ax2.plot(np.log10(np.maximum(0,cost_history)))
ax2.set_ylabel("$log_{10}(mincost)$")
ax2.set_xlabel("$iteration$")
#ax2.set_ylim((-2,1))
plt.show()
