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

import abc
import numpy as np
from scipy.optimize import minimize
from time import time #used for testing
from time import sleep #used for testing


class FitnessFunction:
  """
  Template for definition of fitness function classes.
  Notice that all the methods are "static":
    they do not use the "self" key word in the arguments.
    they do not access object attributes.
  """
  @abc.abstractmethod
  def evaluate(X):
    """
    Outputs the function value at each input in array X.

    X is an array of shape (m,n) where m is the number of inputs and n is the
    number of dimensions.
    """
    return

  @abc.abstractmethod
  def evaluate_single(x):
    """
    Outputs the function value with a single n-dimensional input x

    x is an array of shape (n,) where n is the number of dimensions.
    """
    return

  @abc.abstractmethod
  def default_minimum(ndim):
    """
    Outputs the minimum value of the function WITHIN THE DEFAULT BOUNDS.

    Takes an integer with the number of dimensions.
    Returns a tuple with the minimum value of the function and the vector
      that evaluates to that value.

    Assigning new values to the default bounds which are different from the code
      in this module is not guaranteed to produce correct minimums within the
      new bounds.
    """
    return

  @abc.abstractmethod
  def default_bounds(ndim):
    """
    Outputs vectors with the default lower and upper bounds

    Takes an integer with the number of dimensions.
    Returns a tuple (lb, ub) with vectors of shape (ndim,).
    For a vector x to be within the default bounds, lb[i]<=x[i] and x[i]<=ub[i]
      for i in range(ndim).
    """
    return


class Sphere(FitnessFunction):
  name='Sphere'
  def evaluate(X):
    m,n=X.shape
    return np.sum(X*X,axis=1)
  def evaluate_single(x):
    n,=x.shape
    return np.sum(x*x)
  def default_minimum(ndim):
    return (0.,np.zeros(ndim))
  def default_bounds(ndim):
    return (np.zeros(ndim)-8., np.zeros(ndim)+8)


class Quadric(FitnessFunction):
  name='Quadric'
  def evaluate(X):
    m,n=X.shape
    temp=[ np.sum(X[:,:i+1],axis=1)**2 for i in range(n)]
    return np.sum(temp,axis=0)
  def evaluate_single(x):
    n,=x.shape
    return np.sum([ np.sum(x[:i+1])**2 for i in range(n)])
  def default_minimum(ndim):
    return (0.,np.zeros(ndim))
  def default_bounds(ndim):
    return (np.zeros(ndim)-8., np.zeros(ndim)+8)


class Rastrigin(FitnessFunction):
  name='Rastrigin'
  def evaluate(X):
    m,n=X.shape
    temp=X**2-10*np.cos(2*np.pi*X)+10
    return np.sum(temp,axis=1)
  def evaluate_single(x):
    n,=x.shape
    return np.sum(x**2-10*np.cos(2*np.pi*x)+10)
  def default_minimum(ndim):
    return (0.,np.zeros(ndim))
  def default_bounds(ndim):
    return (np.zeros(ndim)-8., np.zeros(ndim)+8)


class Rosenbrock(FitnessFunction):
  """ https://en.wikipedia.org/wiki/Rosenbrock_function """
  name='Rosenbrock'
  def evaluate(X):
    m,n=X.shape
    assert(n%2==0)
    temp=[100*(X[:,2*i]-X[:,2*i-1]**2)**2+(1-X[:,2*i-1])**2 for i in range(int(n/2))]
    return np.sum(temp,axis=0)
  def evaluate_single(x):
    n,=x.shape
    assert(n%2==0)
    return np.sum([100*(x[2*i]-x[2*i-1]**2)**2+(1-x[2*i-1])**2 for i in range(int(n/2))])
  def default_minimum(ndim):
    return (0.,np.ones(ndim))
  def default_bounds(ndim):
    return (np.zeros(ndim)-8., np.zeros(ndim)+8)


class Schwefel(FitnessFunction):
  """
  http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2530.htm
  http://www.aridolan.com/ofiles/ga/gaa/Schwefel.aspx
  """
  name='Schwefel'
  def evaluate(X):
    m,n=X.shape
    temp=X*np.sin(np.abs(X)**0.5)
    ##original function:
    #return 418.9829*n-np.sum(temp,axis=1)
    #tuned numbers to approach a zero minimum. average error for n in range(100) is about 2e-10
    return 418.98288727224343*n-np.sum(temp,axis=1)
  def evaluate_single(x):
    n,=x.shape
    ##original function:
    #return 418.9829*n-np.sum(x*np.sin(np.abs(x)**0.5))
    #tuned numbers to approach a zero minimum. average error for n in range(100) is about 2e-10
    return 418.98288727224343*n-np.sum(x*np.sin(np.abs(x)**0.5))
  def default_minimum(ndim):
    ##nominal result:
    #return (0., )
    #Numerical result:
    def aux(x):
      return Schwefel.evaluate_single(np.ones(ndim)*x)
    res=minimize(aux,420.9687,tol=1e-20)
    return (res.fun,np.ones(ndim)*res.x)
  def default_bounds(ndim):
    return (np.zeros(ndim)-500., np.zeros(ndim)+500)


class Ackley(FitnessFunction):
  name='Ackley'
  def evaluate(X):
    m,n=X.shape
    return -20*(np.exp(-0.2*((1/n)*np.sum(X*X,axis=1))**0.5)) - np.exp((1/n)*np.sum(np.cos(2*np.pi*X),axis=1))+20+np.e
  def evaluate_single(x):
    n,=x.shape
    return -20*(np.exp(-0.2*((1/n)*np.sum(x*x))**0.5)) - np.exp((1/n)*np.sum([np.cos(2*np.pi*xi) for xi in x]))+20+np.e
  def default_minimum(ndim):
    return (0., np.zeros(ndim))
  def default_bounds(ndim):
    return (np.zeros(ndim)-32., np.zeros(ndim)+32)


class Michalewicz(FitnessFunction):
  name='Michalewicz'
  def evaluate(X):
    m,n=X.shape
    m_number=10
    del m
    temp=[ np.sin(X[:,i])*np.sin((i+1)*(X[:,i]**2)/np.pi)**(2*m_number) for i in range(n)]
    return -np.sum(temp,axis=0)
  def evaluate_single(x):
    n,=x.shape
    m_number=10
    return -np.sum([ np.sin(x[i])*np.sin((i+1)*(x[i]**2)/np.pi)**(2*m_number) for i in range(len(x))])
  def default_minimum(ndim):
    """
    Guess for the Michalewicz function global minimum.

    This an educated guess produced by making a greedy search for each element
     of the input vector individually and sequentially starting from the first.
    It works for ndim=2, 5 and 10 and it has passed heuristical tests with
     thousands of optimized solutions from random start points and up to 100
     dimensions, but I have not been able to prove this always reaches the
     actual global minimum.
    """
    assert(ndim>0)
    e=Michalewicz.evaluate
    e_s=Michalewicz.evaluate_single
    lb,ub=Michalewicz.default_bounds(ndim)

    nsamples=1001
    x0=np.zeros((nsamples,ndim))+(lb+ub)/2.
    for j in range(ndim):
      x0[:,j]=np.linspace(lb[j],ub[j],nsamples)
      y=e(x0)
      x0[:,:]=x0[np.argmin(y),:]
    res=minimize(e_s,x0[0],tol=1e-20)
    return (res.fun,res.x)
  def default_bounds(ndim):
    return (np.zeros(ndim),np.ones(ndim)*np.pi)

class RotatedHyperEllipsoid(FitnessFunction):
  name='RotatedHyperEllipsoid'
  def evaluate(X):
    m,n=X.shape
    ##original version:
    #return np.sum([np.sum(X[:,:i]**2,axis=1) for i in range(1,n+1)],axis=0)
    ##faster version:
    return ((X**2)*np.arange(n,0,-1)).sum(axis=1)
  def evaluate_single(x):
    n,=x.shape
    ##original vesion:
    #return np.sum([np.sum(x[:i]**2) for i in range(1,n+1)])
    ##faster vesion:
    return ((x**2)*np.arange(n,0,-1)).sum()
  def default_minimum(ndim):
    return (0., np.zeros(ndim))
  def default_bounds(ndim):
    return (np.zeros(ndim)-65.536, np.zeros(ndim)+65.536)


def test(c,dims,nsamples,tol):
  """
  Test a subclass of FitnessFunction.

  Takes a class c as an argument, throws an "AssertionError" in case of failure.
  Tests if all the methods are present, if the default minimums reported are
    sane, if the implementations of the evaluation of a single input and an
    array of inputs are equivalent, and randomly tests values and optimizations
    starting from random points to check if a better value is found than the
    reported minimum.
  Also tests if there is any numeric problem evaluating with many dimensions.

  """
  assert(hasattr(c,'evaluate'))
  assert(hasattr(c,'evaluate_single'))
  assert(hasattr(c,'default_bounds'))
  assert(hasattr(c,'default_minimum'))
  assert(hasattr(c,'name'))
  #TODO: test if name is the same as the class name. Use it to report results
  t=time()
  for n in dims:
    #checking for sanity in the number of dimensions:
    assert(n>0) #is the number of dimensions greater than zero?
    assert(n==int(n)) #is the number of dimensions an integer?
    #print a diagnostic if last diagnostic was more than 5 seconds ago.
    if(time()-t>5):

      print(' n='+str(n),'dimensions...')
      t=time()

    ymin,xmin=c.default_minimum(n)
    lb,ub=c.default_bounds(n)
    #is the provided default minimum within the default bounds?
    assert ((xmin>=lb).all())
    assert ((xmin<=ub).all())
    #does the provided default minimum evaluate to the solution (both matrix and single versions)?
    assert (abs(c.evaluate_single(xmin)-ymin)<tol) #test with a tolerance of
    assert (abs(c.evaluate(xmin.reshape(1,-1))-ymin)<tol) #test with a tolerance
    #generate a random test vector between the bounds:
    X=(np.random.random((nsamples,n))-lb)*(ub-lb) #random vector with nsamples values
    #do the single evaluation and the matrix evaluation match?
    se=np.array([c.evaluate_single(x) for x in X]) #single evaluation
    me=c.evaluate(X) #matrix evaluation
    '''
    #DEBUG:
    if not ((np.abs(se-me)<tol).all()):
      offending=np.where(np.abs(se-me)>=tol)
      print( se[offending])
      print( me[offending])
      print( np.abs(se[offending]- me[offending]))
    assert ((np.abs(se-me)<tol).all())
    '''
    assert ((se==me).all())

    #do all the evaluations above produce larger values than the provided minimum?
    assert((me>=ymin).all())
    #is the provided global minimum really minimum?
    res=minimize(c.evaluate_single,xmin,method='L-BFGS-B',bounds=(np.array([lb,ub]).transpose())) #test the built in scipy optimizer at the provided minimum
    if(res.fun<ymin-tol):
      print('better solution found:',res.fun,res.x)
      print(' old solution:',ymin,xmin)
    assert(res.fun>=ymin-tol)
    for x in X:
      res=minimize(c.evaluate_single,x,method='L-BFGS-B',bounds=(np.array([lb,ub]).transpose())) #test the built in scipy optimizer at the random points
      if(res.fun<ymin-tol):
        print('better solution found:',res.fun,res.x)
        print(' old solution:',ymin,xmin)
      assert(res.fun>=ymin-tol)
  return True

all_functions={i[0]:i[1] for i in vars().copy().items() if
                hasattr(i[1],'evaluate') and
                hasattr(i[1],'evaluate_single') and
                hasattr(i[1],'default_bounds') and
                hasattr(i[1],'default_minimum') and
                i[0]!='FitnessFunction'}

def test_all():
  """ Test for all FitnessFunction classes except the FitnessFunction classe. """
  v=all_functions.copy()
  for i in v.keys():
    maxdim=10
    nsamples=100
    print('Testing',i,'with dimensions up to ',maxdim,' and',nsamples,'random samples.')
    if(i=='Rosenbrock'):
      dims=range(2,maxdim+1,2) #rosenbrock only accepts even dimensions
    else:
      dims=range(1,maxdim+1)
    test(v[i],dims,nsamples,1e-9) #test with a tolerance of 1e-9
    print(' ',i,'passed test.')
    plot2dprofile(v[i])
    sleep(2)
  import matplotlib.pyplot as plt
  plt.show()
  return

def plot2dprofile(c,nsamples=101):
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import axes3d
  from matplotlib import cm
  fig = plt.figure(c.name)
  ax=fig.add_subplot(111,projection='3d')
  ((x0,y0),(x1,y1))=c.default_bounds(2)
  zmin,aux=c.default_minimum(2)
  x=np.linspace(x0,x1,nsamples+1)
  y=np.linspace(y0,y1,nsamples+1)
  X,Y=np.meshgrid(x,y)
  inputvectors=np.concatenate(([X.ravel()],[Y.ravel()]),axis=0).transpose()
  Z=c.evaluate(inputvectors).reshape(X.shape)
  #Z=np.array([[(np.array([i,j])) for j in y] for i in x])
  #zmin=np.min(Z)
  #zmax=np.max(Z)
  ax.plot_surface(X,Y,Z,cmap=cm.jet,linewidth=0,rstride=1,cstride=1,antialiased=True)
  ax.contour(X, Y, Z, zdir='z', offset=zmin, cmap=cm.coolwarm)

#plot2dprofile(Michalewicz)
#plot2dprofile(RotatedHyperEllipsoid,201)
#test_all()
