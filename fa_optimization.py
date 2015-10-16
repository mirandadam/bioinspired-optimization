#!/usr/bin/python3
# -*- coding: utf8 -*-
"""
@author: Daniel Araujo Miranda
Licence: Free and open source under the GPL license
Implementation of Xin-She Yang's Firefly Algorithm.

The original MatLab code had the following preamble:

% ======================================================== % 
% Files of the Matlab programs included in the book:       %
% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
% Second Edition, Luniver Press, (2010).   www.luniver.com %
% ======================================================== %    


% =========================================================% 
% Firefly Algorithm by X S Yang (Cambridge University)     %
% Usage: firefly_simple([number_of_fireflies,MaxGeneration])
%  eg:   firefly_simple([12,50]);                          %
% ======================================================== %
% This is a demo for 2D functions; for higher dimenions,   %
% you should use fa_ndim.m or fa_mincon.m                  %
% Parameters choice: 
% Gamma should be linked with scales. Otherwise, the FA    %
% the efficiency will be significantly reduced because     %
% the beta term may be too small.                          %
% Similarly, alpha should also be linked with scales,      %
% the steps should not too large or too small, often       %
% steps are about 1/10 to 1/100 of the domain size.        %
% In addition, alpha should be reduced gradually           %
% using alpha=alpha_0 delta^t during eteration t.          %
% Typically, delta=0.9 to 0.99 will be a good choice.      %
% ======================================================== %
"""

import numpy as np
import fitnessfunctions

#Firefly parameters:
u=8
l=-8
N=6
default_parameters={
  #Algorithm tuning:
  'S': 20, #Swarm size
  'alpha': 1*((u-l)/20), #coefficient of random movement, should be related to the range of possible values. 1/100 of the full range
  'delta': 0.99, #randomization dampening coefficient
  'exponent': (1/(u-l))**0.5, #gamma, light absorption exponent, should be related to the range of possible values. 1/(range^0.5)
  'beta0': 1, #constant that multiplies the attraction

  #Problem parameters:
  'target_cost': -1,     #Stopping condition. If a solution reaches this cost or lower the algorithm halts.
  'N': N, #number of dimensions
  'ub': np.ones(N)*u,    #lower bounds of the parameters
  'lb': np.ones(N)*l,  #upper bounds of the parameters  
  'maxiter':500,          #Stopping condition. Maximum number of iterations to try.
}


def fa(f=fitnessfunctions.rosenbrock_m, parameters=default_parameters):
  global default_parameters
  p=default_parameters.copy()
  p.update(parameters)
  S=p['S'] #Swarm size
  alpha=p['alpha'] #coefficient of random movement
  delta=p['delta'] #randomization dampening coefficient
  exponent=p['exponent'] #light absorption coefficient
  beta0=p['beta0']
  
  target_cost=p['target_cost'] #Stopping condition. If a solution reaches this cost or lower the algorithm halts.
  N=p['N'] #number of dimensions
  ub=p['ub'] #lower bounds of the parameters
  lb=p['lb'] #upper bounds of the parameters  
  maxiter=p['maxiter'] #Stopping condition. Maximum number of iterations to try.
  
  del p, parameters
  
  #aux_ub=np.array([ub]).repeat(S,axis=0)
  #aux_lb=np.array([lb]).repeat(S,axis=0)
  #X=aux_lb+np.random.rand(S,N)*(aux_ub-aux_lb)
  #del aux_ub, aux_lb
  X=lb+np.random.rand(S,N)*(ub-lb)
  
  Y=f(X)
  sequence=np.argsort(Y)
  min_cost=Y[sequence[0]]
  
  cost_history=[]
  X_history=[]
  
  cost_history.append(min_cost)
  X_history.append(X.copy())
  
  for i in range(maxiter):
    nextX=X+alpha*(np.random.rand(S,N)-0.5) #beta term is calculated on the loop
    for j in range(1,S):
      #calculation of the attraction (beta) contribution:
      d=X[sequence[:j]]-X[sequence[j]]
      r2=(d*d).sum(axis=1)
      beta=beta0*np.exp(-exponent*r2).reshape((-1,1))
      nextX[sequence[j]]+=(beta*d).sum(axis=0)
    del d,r2,beta
    nextX=np.minimum(ub,nextX)
    nextX=np.maximum(lb,nextX)
    nextY=f(nextX)
    #WARNING!!! firefly algorithm does not seem to check if individual solutions were improved
    #aux=np.where(nextY<Y)    
    #X[aux]=nextX[aux]
    #Y[aux]=nextY[aux]
    X=nextX.copy()
    Y=nextY.copy()
    del nextX,nextY
  
    sequence=np.argsort(Y)
    min_cost=Y[sequence[0]]
    cost_history.append(min_cost)
    X_history.append(X.copy())
    
    alpha=alpha*delta
    if(min_cost<=target_cost):
      break
  
  return X_history,cost_history

#'''
#quick test:
#f,u,l=fitnessfunctions.sphere_m,8,-8
#f,u,l=fitnessfunctions.schwefel_m,500,-500
#f,u,l=fitnessfunctions.rastrigin_m,8,-8
f,u,l=fitnessfunctions.ackley_m,32,-32
#f,u,l=fitnessfunctions.michalewicz_m,np.pi,0

N=2
S=20
X_history,cost_history=fa( f,
                 {'target_cost':-10000,
                  'maxiter':1000,
                  'N':N,
                  'S':S,
                  'alpha': 1*(u-l)/20,
                  'delta': 0.99,
                  'exponent':(1/(u-l))**0.5,
                  'beta0':1,
                  'ub': np.ones(N)*u,    #lower bounds of the parameters
                  'lb': np.ones(N)*l,  #upper bounds of the parameters
                  })

from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

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

if(N==2):
  fig_a=plt.figure()
  ax=fig_a.add_subplot(111,projection='3d')
  nsamples=100
  x0=np.linspace(l,u,nsamples+1)
  x1=np.linspace(l,u,nsamples+1)
  X0,X1=np.meshgrid(x0,x1)
  X=np.array([X0.ravel(),X1.ravel()]).transpose()
  Z0=f(X).reshape(X0.shape)
  zmin=np.min(Z0)
  zmax=np.max(Z0)
  #ax.plot_surface(X0,X1,Z0,cmap=cm.jet,linewidth=0,rstride=1,cstride=1,antialiased=True)
  ax.contour(X0, X1, Z0, zdir='z', offset=zmin-0.1*(zmax-zmin), cmap=cm.coolwarm)
  
  s0=ax.scatter(X_history[0][:,0],X_history[0][:,1],f(X_history[0]),color='g')
  s1=ax.scatter(X_history[0][:,0],X_history[0][:,1],np.zeros(S),color='b')
  
  def update_position(num,data,plots,f):
    #s0,s1=plots
    s0,s1=plots
    hx=data
    s0._offsets3d=[hx[num][:,0],hx[num][:,1],f(hx[num])]
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