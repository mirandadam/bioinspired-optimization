import numpy as np

def sphere(x):
  #r=numpy.array(x).ravel() #needed if input is unbehaved
  return np.sum(x*x)

def quadric(x):
  #r=numpy.array(x).ravel() #needed if input is unbehaved
  N=len(x)
  return np.sum([ np.sum([x[:i+1]])**2 for i in range(N)])

twopi=2*np.pi
def rastrigin(x):
  N=len(x)
  return np.sum([xi**2-10*np.cos(twopi*xi)+10 for xi in x])

def rosenbrock(x): 
  N=len(x)
  return np.sum([100*(x[2*i]-x[2*i-1]**2)**2+(1-x[2*i-1])**2 for i in range(int(N/2))])

def schwefel(x):
  #suggested interval: [-500,500]
  N=len(x)
  return 418.9829*N-np.sum([xi*np.sin(np.abs(xi)**0.5) for xi in x])

def ackley(x):
  #suggested interval: [-32,32]
  N=len(x)
  return -20*(np.exp(-0.2*((1/N)*np.sum(x*x))**0.5)) - np.exp((1/N)*np.sum([np.cos(twopi*xi) for xi in x]))+20+np.e

def michalewicz(x):
  #suggested interval: [0, pi]
  m=10
  return -np.sum([ np.sin(x[i])*np.sin((i+1)*(x[i]**2)/np.pi)**(2*m) for i in range(len(x))]);

'''
#testing:
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
  ax.plot_surface(X,Y,Z,cmap=cm.jet,linewidth=0,rstride=1,cstride=1,antialiased=True)
  cset = ax.contour(X, Y, Z, zdir='z', offset=0, cmap=cm.coolwarm)

#generate_surface(esfera,-5,-5,5,5,101)
#generate_surface(quadric,-5,-5,5,5,101)
#generate_surface(rastrigin,-3,-3,3,3,101)
#generate_surface(rosenbrock,-3,-3,3,3,101)
#generate_surface(schwefel,-500,-500,500,500,101)
#generate_surface(ackley,-10,-10,10,10,101)
generate_surface(michalewicz,0,0,np.pi,np.pi,101)
plt.show()
'''
