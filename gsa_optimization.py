#!/usr/bin/python3
# -*- coding: utf8 -*-
"""
@author: Daniel Araujo Miranda
Licence: Free and open source under the GPL license
"""


'''
Implementation of the Gravitational Search Algorithm for optimization as 
described in :
  GSA: A Gravitational Search Algorithm
  Rashedi E., Nezamabadi-pour H., Saryazdi S.
  (2009) Information Sciences, 179 (13) , pp. 2232-2248.
'''

import numpy as np
import fitnessfunctions



#a) search space identification
f,u,l=fitnessfunctions.rosenbrock_m,8,-8
#f,u,l=fitnessfunctions.sphere_m,8,-8
#f,u,l=fitnessfunctions.quadric_m,8,-8
#f,u,l=fitnessfunctions.schwefel_m,500,-500
#f,u,l=fitnessfunctions.rastrigin_m,8,-8
#f,u,l=fitnessfunctions.ackley_m,32,-32
#f,u,l=fitnessfunctions.michalewicz_m,np.pi,0

S=50
N=2
maxiter=500

#PSO:
#w_initial=0.9
#w_final=0.2

#GSA:
G0=100
#G_decay=0.99 #G(t+1)=G(t)*G_decay #slightly different from the formulation in the article, which seems to grow with time instead of decreasing
alpha=20;
epsilon=0.0001 #small value to be added to the distance in the denominator to avoid singularities

#b) randomized initialization
X=(np.random.rand(S,N)*(u-l) + l)
V=np.zeros(X.shape)
#c) fitness evaluation of agents
Y=f(X)
sequence=np.argsort(Y)
#d) update G(t), best(t), worst(t) and Mi(t)
bestY=Y.min()
worstY=Y.max()

cost_history=[bestY]
X_history=[X.copy()]
for i in range(maxiter):
  #d) update G(t), best(t), worst(t) and Mi(t)
  G=G0*np.exp(-alpha*i/maxiter)
  #kbest=int( 1+(S-1)*(maxiter-i)/maxiter )
  kbest=int(round( (0.02+(1-i/maxiter)*(1-0.02))*S ))
  #article method to calculate m:
  m=(Y-worstY)/(bestY-worstY+epsilon) #calculate the inertial mass between 0 and 1
  #m=1-(Y-bestY)/(worstY-bestY)
  M=m/np.maximum(epsilon,m.sum()) #make the total system mass equal 1
  M=M.reshape((-1,1)) #transpose the matrix
  #e) Calculation of the total force in different directions
  A=np.zeros(X.shape)
  for j in range(kbest):
    #D=X[sequence[j]]-X #vector distance between all and the Xj attracting element
    D=X[sequence[j]]-X #vector distance between all and the Xj attracting element
    r=((D*D).sum(axis=1))**0.5 #scalar distance
    r=r.reshape((-1,1)) #transpose the matrix
    #merging equations 7, 9 and 10:
    A=A+np.random.rand(S,1)*G*M[sequence[j]]*D/(r+epsilon)  #adding the force contributed by the Xj body
  #merging equations 10 and 11:
  V=np.random.rand(S,1)*V+A
  X=X+V
  X=np.maximum(X,l)
  X=np.minimum(X,u)
  #c) fitness evaluation of agents
  Y=f(X)
  sequence=np.argsort(Y)
  #d) update G(t), best(t), worst(t) and Mi(t)
  bestY=Y.min()
  worstY=Y.max()
  cost_history.append(bestY)
  X_history.append(X.copy())
  if(i%20==0):
    print(kbest)


'''
#quick test:
#f,u,l=fitnessfunctions.sphere_m,8,-8
#f,u,l=fitnessfunctions.schwefel_m,500,-500
#f,u,l=fitnessfunctions.rastrigin_m,8,-8
#f,u,l=fitnessfunctions.ackley_m,32,-32

N=6
S=20
X_history,cost_history=fa( f,
                 {'target_cost':-1,
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
#'''

#'''
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
ax2.plot(np.log10(np.maximum(1e-20,cost_history)))
ax2.set_ylabel("$log_{10}(mincost)$")
ax2.set_xlabel("$iteration$")
#ax2.set_ylim((-2,1))

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
  ax.plot_surface(X0,X1,Z0,cmap=cm.jet,linewidth=0,rstride=1,cstride=1,antialiased=True)
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