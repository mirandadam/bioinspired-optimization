#!/usr/bin/python3
# -*- coding: utf8 -*-

import numpy as np
import base

class MOPSO: #multi-objective PSO
  def __init__(self,cost_function,n_dimensions,n_objectives,lb,ub,maxiter=500,
               population_size=80,
               initial_inertia=0.9,
               final_inertia=0.1,
               social_coefficient=2.1,
               cognitive_coefficient=2.1,
               repulsion_coefficient=0.1, #other population members repel this one
               maximum_velocity=0.1#,#ub-lb/10
               #initial_velocity=0.01 #maximum_velocity/10
               ):
    s=self
    s.cost_function=cost_function
    s.n_dimensions=n_dimensions
    s.n_objectives=n_objectives
    s.lb=lb.copy()
    s.ub=ub.copy()
    s.maxiter=maxiter

    s.population_size=population_size
    s.initial_inertia=initial_inertia
    s.final_inertia=final_inertia
    s.inertia_decay_rate=(final_inertia/initial_inertia)**(1/maxiter)
    s._w=initial_inertia
    s.social_coefficient=social_coefficient
    s.cognitive_coefficient=cognitive_coefficient
    s.repulsion_coefficient=repulsion_coefficient
    s.maximum_velocity=maximum_velocity #ub-lb/10
    #s.initial_velocity=initial_velocity #maximum_velocity/10

    #initial conditions:
    s._X = np.random.random((population_size,n_dimensions))*(ub-lb)+lb # current individuals
    s._Y = s.cost_function(s._X)                        # current cost
    s._V = s._X*0.
    assert(s._Y.shape[1]==n_objectives)

    rank=base.nonDominatedRankV2(s._Y)
    aux=np.where(rank==1)[0]
    s._global_bestX=s._X[aux]
    s._global_bestY=s._Y[aux]
    s._individual_bestX=s._X.copy()
    s._individual_bestY=s._Y.copy()
    s._iter=0


  def iterate_one(self):
    s=self
    #calculating inertia weight:
    s._w=s._w*s.inertia_decay_rate
    w=s._w
    #particle movement:
    #nearest_global_best=np.zeros((s.population_size,s.n_dimensions)) #distance in fitness space
    #nearest_projected_best=np.zeros((s.population_size,s.n_dimensions)) #virtual point which ponders the nearest ones to guess the nearest one on the pareto border
    nearest_neighbour=np.zeros((s.population_size,s.n_dimensions))   #distance in fitness space
    for i in range(s.population_size):
      ##getting nearest global best
      #aux=(((s._global_bestY-s._Y[i])**2).sum(axis=1))**0.5
      #aux2=np.where(aux==np.min(aux))[0][0]
      #temp=s._global_bestX[aux2]
      #nearest_global_best[i]=s._global_bestX[aux2]

      ##getting nearest projected best:
      #aux=(((s._global_bestY-s._Y[i])**2).sum(axis=1))**0.5
      #aux2=np.argsort(aux)[:s.n_objectives] #get as many points as necessary to interpolate the pareto front nearest to the point
      #if(aux[aux2[0]]==0):
      #  nearest_projected_best[i]=s._global_bestX[aux2[0]]
      #else:
      #  nearest_projected_best[i]=(s._global_bestX[aux2]*((aux[aux2]**-1).reshape(-1,1))/np.sum(aux[aux2]**-1)).sum(axis=0)

      #getting nearest neighbour
      aux=(((s._Y-s._Y[i])**2).sum(axis=1))**0.5
      aux[i]=np.inf
      aux2=np.where(aux==np.min(aux))[0][0]
      nearest_neighbour[i]=s._X[aux2]    
    
    #if(s._iter%5==0):
    #  o_global=base.component_sort_order(s._global_bestY)
    #  o_current=base.component_sort_order(s._Y)
    #  o_global=o_global*len(o_current)/len(o_global) #make sure the range is the same
    #  s.nearest_global_best_by_order=np.zeros((s.population_size,s.n_dimensions)) #distance in fitness space
    #  taken=np.zeros(len(o_global),'bool')
    #  for i in range(s.population_size):
    #    #getting nearest by order:
    #    aux=(((o_global-o_current[i])**2).sum(axis=1))**0.5
    #    aux2=np.argsort(aux)
    #    for j in aux2:
    #      if not taken[j]:
    #        break
    #    s.nearest_global_best_by_order[i]=s._global_bestX[j]
    #    taken[j]=True
    
    if(s._iter%5==0):
      aux=len(s._global_bestX)*(np.arange(s.population_size))/s.population_size
      #s.designated_target=s._global_bestX[aux.astype('int')]
      s.designated_target=s._global_bestX[aux[np.random.permutation(len(aux))].astype('int')]
    
    r1=np.random.random((s.population_size,s.n_dimensions))
    r2=np.random.random((s.population_size,s.n_dimensions))
    r3=np.random.random((s.population_size,s.n_dimensions))
    #r4=np.random.random((s.population_size,s.n_dimensions))-0.5
    s._V= w*s._V + \
          s.cognitive_coefficient*r1*(s._individual_bestX-s._X) +\
          s.social_coefficient*r2*(s.designated_target-s._X) +\
          - r3*s.repulsion_coefficient*(nearest_neighbour-s._X) #+\
          #r4*0.00001
          #s.social_coefficient*r2*(s.nearest_global_best_by_order-s._X) +\
          #
          #s.social_coefficient*r2*(nearest_projected_best-s._X) +\
          #- r3*s.repulsion_coefficient*(nearest_neighbour-s._X)# +\
          #s.social_coefficient*r2*(nearest_global_best-s._X) + \
 
    #applying speed limit:
    vnorm=((s._V**2).sum(axis=1))**0.5 #norm of the speed
    aux=np.where(vnorm>s.maximum_velocity) #particles with velocity greater than expected
    s._V[aux]=s._V[aux]*s.maximum_velocity/(vnorm[aux].reshape((-1,1))) #clipping the speed to the maximum speed

    #update solutions:
    s._X=s._X+s._V

    #clipping the search space
    s._X=np.minimum(s.ub,s._X)
    s._X=np.maximum(s.lb,s._X)

    #fitness value calculation:
    s._Y = s.cost_function(s._X)  # current particle cost

    #update memories:
    temp_Y=np.concatenate((s._global_bestY,s._Y),axis=0).copy()
    temp_X=np.concatenate((s._global_bestX,s._X),axis=0).copy()
    Y,rank,cf,indices=base.truncate(temp_Y,1*s.population_size)
    #s._global_bestX=temp_X[indices][np.where(rank==1)[0]]
    #s._global_bestY=temp_Y[indices][np.where(rank==1)[0]]
    s._global_bestX=temp_X[indices]
    s._global_bestY=temp_Y[indices]
    

    #debugging code:
    #for i in s._global_bestY:
    #  assert((temp_Y>=i).any(axis=1).all(axis=0))
    #print('len_global',len(s._global_bestX),len(s._global_bestY))
    
    #updating individual best:
    for i in range(s.population_size):
      if (s._individual_bestY[i]>=s._Y[i]).all() and (s._individual_bestY[i]>s._Y[i]).any():
        #new solution dominates old one
        s._individual_bestY[i]=s._Y[i]
        s._individual_bestX[i]=s._X[i]
      elif (s._individual_bestY[i]<s._Y[i]).any():
        #new solution not dominated by the old one. See if it approaches the pareto front best
        #'''
        aux=(((s._global_bestY-s._Y[i])**2).sum(axis=1))**0.5
        aux2=np.argsort(aux)[:s.n_objectives] #get as many points as necessary to interpolate the pareto front nearest to the point
        if(aux[aux2[0]]==0):
          aux3=s._global_bestY[aux2[0]]
        else:
          #average the nearby points weighted by the inverse distance:
          aux3=(s._global_bestY[aux2]*((aux[aux2]**-1).reshape(-1,1))/np.sum(aux[aux2]**-1)).sum(axis=0)
        new_min=((aux3-s._Y[i])**2).sum()**0.5

        aux=(((s._global_bestY-s._individual_bestY[i])**2).sum(axis=1))**0.5
        aux2=np.argsort(aux)[:s.n_objectives] #get as many points as necessary to interpolate the pareto front nearest to the point
        if(aux[aux2[0]]==0):
          aux3=s._global_bestY[aux2[0]]
        else:
          #average the nearby points weighted by the inverse distance:
          aux3=(s._global_bestY[aux2]*((aux[aux2]**-1).reshape(-1,1))/np.sum(aux[aux2]**-1)).sum(axis=0)
        old_min=((aux3-s._individual_bestY[i])**2).sum()**0.5
        #'''
        '''
        aux=(((s._global_bestY-s._Y[i])**2).sum(axis=1))**0.5
        new_min=np.min(aux)
        aux=(((s._global_bestY-s._individual_bestY[i])**2).sum(axis=1))**0.5
        old_min=np.min(aux)
        '''
        if(old_min>new_min):
          s._individual_bestY[i]=s._Y[i]
          s._individual_bestX[i]=s._X[i]

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


