#!/usr/bin/python3
# -*- coding: utf8 -*-
"""
@author: Daniel Araujo Miranda
Licence: Free and open source under the GPL license
"""

import numpy as np
import fitnessfunctions

#default Particle Swarm Optimization (PSO) parameters:
default_parameters={
  'S':10,                #number of particles
  'dimensions':6,                 #number of dimensions
  'maxiter':100, #1000   #maximum number of iterations
  'w0':0.9,              #initial inertia coefficient (weight)
  'wf':0.1,              #final inertia coefficient (weight)
  'c1':2,                #cognitive coefficient
  'c2':2,                #social coefficient
  'max_v':5,             #maximum velocity
  'ini_v':5/10,#max_v/10 #initial velocity
  'x_max':8,             #range domain for fitness function - inferior limit (assumes the same for all dimensions)
  'x_min':-8,            #range domain for fitness function - superior limit (assumes the same for all dimensions)
  'target_cost':1e-10
}


def pso(f,parameters):
  global default_parameters
  p=default_parameters.copy()
  p.update(parameters)
  S=p['S']
  dimensions=p['dimensions']
  maxiter=p['maxiter']
  w0=p['w0']
  wf=p['wf']
  c1=p['c1']
  c2=p['c2']
  max_v=p['max_v']
  ini_v=p['ini_v']
  x_max=p['x_max']
  x_min=p['x_min']
  target_cost=p['target_cost']

  x=np.random.random((S,dimensions))*(x_max-x_min)+x_min
  y=np.ones((S,dimensions),dtype='float64')*1e10
  v=np.ones((S,dimensions),dtype='float64')*ini_v
  f_ind=np.ones(S,dtype='float64')*1e10 # initialize best local fitness 
  cost_history=[]
  X_history=[]
  

  w= w0 #current weight
  slope = (wf-w0)/maxiter #increment that will take w from w0 to wf in the specified number of iteractions
  k = 1
  while k<=maxiter:
    #fitness value calculation:
    fx=f(x)#np.array([f(i) for i in x])
    
    #memory update:
    aux=np.where(fx<f_ind)
    f_ind[aux]=fx[aux]
    y[aux]=x[aux]
    best=np.min(f_ind)
    aux=np.where(f_ind==best)[0][0]
    cost_history.append(best)
    X_history.append(x.copy())
    ys=y[aux]
    
    #test if target cost was reached:
    if(best<=target_cost):
      break

    #particle movement:  
    r1=np.random.random((S,dimensions))
    r2=np.random.random((S,dimensions))
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

  #print(ys)
  #print(best)
  return(cost_history,X_history)

#'''
#quick test:
cost_history,X_history=pso(fitnessfunctions.ackley_m,
                {'x_min':-32,
                'x_max':32,
                'dimensions':2,
                'S':30,
                'maxiter':1000,
                'target_cost':-1})
N=2
f=fitnessfunctions.ackley_m
l=-32
u=32
S=30

#cost_history=pso(fitnessfunctions.michalewicz_m,
#                {'x_min':0,
#                'x_max':np.pi,
#                'dimensions':6,
#                'S':30,
#                'maxiter':1000,
#                'target_cost':-10})
                
#cost_history=pso(fitnessfunctions.schwefel_m,
#                {'x_min':-500,
#                'x_max':500,
#                'dimensions':6,
#                'S':30,
#                'maxiter':1000,
#                'target_cost':0.01})
                
#cost_history=pso(fitnessfunctions.rosenbrock_m,
#                {'x_min':-8,
#                'x_max':8,
#                'dimensions':6,
#                'S':30,
#                'maxiter':10000})

from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

fig = plt.figure()
ax1=fig.add_subplot(211)
ax1.plot(cost_history)
ax1.set_ylabel("$mincost$")
if(np.min(cost_history)>0):
  ax2=fig.add_subplot(212)
  ax2.plot(np.log10(cost_history))
  ax2.set_ylabel("$log_{10}(mincost)$")
  ax2.set_xlabel("$iteration$")
plt.show()


if(N==2):
  fig_a=plt.figure()
  ax=fig_a.add_subplot(111,projection='3d')
  nsamples=200
  x0=np.linspace(l,u,nsamples+1)
  x1=np.linspace(l,u,nsamples+1)
  X0,X1=np.meshgrid(x0,x1)
  X=np.array([X0.ravel(),X1.ravel()]).transpose()
  Z0=f(X).reshape(X0.shape)
  zmin=np.min(Z0)
  zmax=np.max(Z0)
  #ax.plot_surface(X0,X1,Z0,cmap=cm.jet,linewidth=0,rstride=1,cstride=1,antialiased=True)
  ax.contour(X0, X1, Z0, zdir='z', offset=zmin-0.0*(zmax-zmin), cmap=cm.coolwarm)
  
  s0=ax.scatter(X_history[0][:,0],X_history[0][:,1],f(X_history[0])+(u-l)/100,color='g')
  s1=ax.scatter(X_history[0][:,0],X_history[0][:,1],np.zeros(S),color='b')
  
  def update_position(num,data,plots,f):
    #s0,s1=plots
    s0,s1=plots
    hx=data
    s0._offsets3d=[hx[num][:,0],hx[num][:,1],f(hx[num])+(u-l)/100]
    s1._offsets3d=[hx[num][:,0],hx[num][:,1],np.zeros(len(hx[num]))]
    print(num)
  
  
  an = animation.FuncAnimation(fig_a,
                               update_position,
                               frames=len(X_history),
                               interval=100,
                               fargs=(X_history,(s0,s1),f)
                               )
plt.show()

#'''

