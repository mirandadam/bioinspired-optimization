#!/usr/bin/python3
# -*- coding: utf8 -*-
"""
@author: Daniel Araujo Miranda
Licence: Free and open source under the GPL license
"""

import numpy as np

def sphere(x):
  #suggested interval: [-8,8]
  #r=numpy.array(x).ravel() #needed if input is unbehaved
  return np.sum(x*x)

def sphere_m(X):
  #Matrix version
  return np.sum(X*X,axis=1)
  
def test_sphere():
  X=16*np.random.rand(20,6)-8
  c=np.array([sphere(x) for x in X])
  cm=sphere_m(X)
  return (np.max(np.abs(cm-c))<1e-100)

def quadric(x):
  #suggested interval: [-8,8]
  #r=numpy.array(x).ravel() #needed if input is unbehaved
  N=len(x)
  return np.sum([ np.sum(x[:i+1])**2 for i in range(N)])

def quadric_m(X):
  #suggested interval: [-8,8]
  #r=numpy.array(x).ravel() #needed if input is unbehaved
  N=X.shape[1]
  temp=[ np.sum(X[:,:i+1],axis=1)**2 for i in range(N)]
  return np.sum(temp,axis=0)

def test_quadric():
  X=16*np.random.rand(20,6)-8
  c=np.array([quadric(x) for x in X])
  cm=quadric_m(X)
  return (np.max(np.abs(cm-c))<1e-100)

def rastrigin(x):
  #suggested interval: [-8,8]
  N=len(x)
  return np.sum([xi**2-10*np.cos(2*np.pi*xi)+10 for xi in x])
  
def rastrigin_m(X):
  #suggested interval: [-8,8]
  N=X.shape[1]
  temp=X**2-10*np.cos(2*np.pi*X)+10
  return np.sum(temp,axis=1)
  
def test_rastrigin():
  X=16*np.random.rand(20,6)-8
  c=np.array([rastrigin(x) for x in X])
  cm=rastrigin_m(X)
  return (np.max(np.abs(cm-c))<1e-100)

def rosenbrock(x): 
  #suggested interval: [-8,8]
  N=len(x)
  return np.sum([100*(x[2*i]-x[2*i-1]**2)**2+(1-x[2*i-1])**2 for i in range(int(N/2))])
  
def rosenbrock_m(X):
  N=X.shape[1]
  temp=[100*(X[:,2*i]-X[:,2*i-1]**2)**2+(1-X[:,2*i-1])**2 for i in range(int(N/2))]
  return np.sum(temp,axis=0)

def test_rosenbrock():
  X=16*np.random.rand(20,6)-8
  c=np.array([rosenbrock(x) for x in X])
  cm=rosenbrock_m(X)
  return (np.max(np.abs(cm-c))<1e-100)
  
def schwefel(x):
  #suggested interval: [-500,500]
  N=len(x)
  return 418.9829*N-np.sum([xi*np.sin(np.abs(xi)**0.5) for xi in x])
  
def schwefel_m(X):
  #suggested interval: [-500,500]
  N=X.shape[1]
  temp=X*np.sin(np.abs(X)**0.5)
  return 418.9829*N-np.sum(temp,axis=1)

def test_schwefel():
  X=1000*np.random.rand(20,6)-500
  c=np.array([schwefel(x) for x in X])
  cm=schwefel_m(X)
  return (np.max(np.abs(cm-c))<1e-100)

def ackley(x):
  #suggested interval: [-32,32]
  N=len(x)
  return -20*(np.exp(-0.2*((1/N)*np.sum(x*x))**0.5)) - np.exp((1/N)*np.sum([np.cos(2*np.pi*xi) for xi in x]))+20+np.e

def ackley_m(X):
  #suggested interval: [-32,32]
  N=X.shape[1]
  temp=-20*(np.exp(-0.2*((1/N)*np.sum(X*X,axis=1))**0.5)) - np.exp((1/N)*np.sum(np.cos(2*np.pi*X),axis=1))+20+np.e
  return temp
  
def test_ackley():
  X=64*np.random.rand(20,6)-32
  c=np.array([ackley(x) for x in X])
  cm=ackley_m(X)
  return (np.max(np.abs(cm-c))<1e-100)
  
def michalewicz(x):
  #suggested interval: [0, pi]
  m=10
  return -np.sum([ np.sin(x[i])*np.sin((i+1)*(x[i]**2)/np.pi)**(2*m) for i in range(len(x))]);
  
def michalewicz_m(X):
  #suggested interval: [0, pi]
  N=X.shape[1]
  m=10
  temp=[ np.sin(X[:,i])*np.sin((i+1)*(X[:,i]**2)/np.pi)**(2*m) for i in range(N)]
  return -np.sum(temp,axis=0);

def test_michalewicz():
  X=np.pi*np.random.rand(20,6)-0
  c=np.array([michalewicz(x) for x in X])
  cm=michalewicz_m(X)
  return (np.max(np.abs(cm-c))<1e-100)

if not test_sphere():
  print("ERROR: Self test failed for function "+"sphere")
  
if not test_quadric():
  print("ERROR: Self test failed for function "+"quadric")
  
if not test_rastrigin():
  print("ERROR: Self test failed for function "+"rastrigin")
  
if not test_rosenbrock():
  print("ERROR: Self test failed for function "+"rosenbrock")
  
if not test_schwefel():
  print("ERROR: Self test failed for function "+"schwefel") 

if not test_ackley():
  print("ERROR: Self test failed for function "+"ackley") 

if not test_michalewicz():
  print("ERROR: Self test failed for function "+"michalewicz") 

def benchmark_all():
  from time import time
  N=22
  S=30
  maxiter=1000
  functions=[
    [sphere,sphere_m,'sphere',[-8,8]],
    [quadric,quadric_m,'quadric',[-8,8]],
    [rastrigin,rastrigin_m,'rastrigin',[-8,8]],
    [rosenbrock,rosenbrock_m,'rosenbrock',[-8,8]],
    [schwefel,schwefel_m,'schwefel',[-500,500]],
    [ackley,ackley_m,'ackley',[-32,32]],
    [michalewicz,michalewicz_m,'michalewicz',[0,np.pi]]
    ]
  for func in functions:      
    X=(func[3][1]-func[3][0])*np.random.rand(S,N)+func[3][0]
    f=func[0]
    fm=func[1]
    t0=time()
    for i in range(maxiter):
      c=np.array([f(x) for x in X])
    t1=time()
    t=t1-t0
    t0=time()
    for i in range(maxiter):
      c=fm(X)
    t1=time()
    tm=t1-t0
    print(func[2],'regular:',t,'matrix:',tm,'speedup:',t/tm)


'''
#graphical testing:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

def generate_surface(f,x0,y0,x1,y1,nsamples):
  fig = plt.figure()
  ax=fig.add_subplot(111,projection='3d')
  x=np.linspace(x0,x1,nsamples+1)
  y=np.linspace(y0,y1,nsamples+1)
  X,Y=np.meshgrid(x,y)
  Z=np.array([[f(np.array([i,j])) for j in y] for i in x])
  zmin=np.min(Z)
  zmax=np.max(Z)
  ax.plot_surface(X,Y,Z,cmap=cm.jet,linewidth=0,rstride=1,cstride=1,antialiased=True)
  cset = ax.contour(X, Y, Z, zdir='z', offset=zmin-0.1*(zmax-zmin), cmap=cm.coolwarm)

#generate_surface(sphere,-5,-5,5,5,101)
#generate_surface(quadric,-5,-5,5,5,101)
#generate_surface(rastrigin,-3,-3,3,3,101)
#generate_surface(rosenbrock,-3,-3,3,3,101)
generate_surface(schwefel,-500,-500,500,500,201)
#generate_surface(ackley,-10,-10,10,10,101)
#generate_surface(michalewicz,0,0,np.pi,np.pi,101)
plt.show()
#'''
