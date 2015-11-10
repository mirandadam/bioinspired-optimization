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

#####WARNING!!!!!!!!! RIGHT NOW THIS CODE IS A MESS AND POORLY TESTED!!!! #############
#TODO: refactor this code.

import numpy as np
import pickle

import sys
sys.path.append('../../src_python') #adding the folder to the import path
import bio #only works because of the two lines above.

#original dataset at http://archive.ics.uci.edu/ml/datasets/Iris
#loading data:
f=open('iris_data/bezdekIris.data','r')
#reading lines and splitting fields:
d=[i.strip('\r\n\t ').split(',') for i in f.readlines() if ',' in i]
f.close()
del f

#collecting all flower names:
names=list(set([i[-1] for i in d])) #making a list with all the unique names
aux=np.identity(len(names)) #auxiliary identity square matrix. each line represents a class. numbers are "almost one" so they are reachable by the sigmoid function with zero error
classes={names[i]:aux[i] for i in range(len(names))} #dictionary linking flower names with a class vector
"""
Variable values at this point:
names=['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']
aux=np.array([[ 1.,  0.,  0.],
              [ 0.,  1.,  0.],
              [ 0.,  0.,  1.]])
classes={'Iris-virginica':  array([ 1.,  0.,  0.]),
         'Iris-versicolor': array([ 0.,  1.,  0.]),
         'Iris-setosa':     array([ 0.,  0.,  1.])}
"""
del names,aux #clearing up unused variables

inputs=np.array([list(map(float,i[:-1])) for i in d]) #inputs from the training set
targets=np.array([classes[i[-1]] for i in d]) #target outputs for each input
del d #clearing up unused variables

"""
First five values of "inputs":
array([[ 5.1,  3.5,  1.4,  0.2],
       [ 4.9,  3. ,  1.4,  0.2],
       [ 4.7,  3.2,  1.3,  0.2],
       [ 4.6,  3.1,  1.5,  0.2],
       [ 5. ,  3.6,  1.4,  0.2]])

First five values of "targets":
array([[ 0.,  0.,  1.],
       [ 0.,  0.,  1.],
       [ 0.,  0.,  1.],
       [ 0.,  0.,  1.],
       [ 0.,  0.,  1.]])

"""

#normalizing inputs:
m=inputs.mean(axis=0)
s=inputs.std(axis=0)
inputs=(inputs-m)/s

"""
attribute average values:
m=np.array([ 5.84333333,  3.05733333,  3.758     ,  1.19933333])
standard deviation of each attribute:
s=np.array([ 0.82530129,  0.43441097,  1.75940407,  0.75969263])

normalized first five inputs:
array([[-0.90068117,  1.01900435, -1.34022653, -1.3154443 ],
       [-1.14301691, -0.13197948, -1.34022653, -1.3154443 ],
       [-1.38535265,  0.32841405, -1.39706395, -1.3154443 ],
       [-1.50652052,  0.09821729, -1.2833891 , -1.3154443 ],
       [-1.02184904,  1.24920112, -1.34022653, -1.3154443 ]])
"""


#### Neural network parameters: ####
input_neurons=inputs.shape[1]    #these neurons are all linear and connect the inputs to the intermediary layer
hidden_neurons=3                 #defines the number of neurons in the intermediary layer.
output_neurons=targets.shape[1]  #the number of output neurons

#total number of input variables:
N=(input_neurons+output_neurons+1)*hidden_neurons+output_neurons


def propagate(inputs,X):
  """
     Propagates "inputs" through the neural network with parameters "X" and returns the outputs.

     Depends on external variables: input_neurons, hidden_neurons and output_neurons.

     first input_neurons*hidden_neurons elements are connecting weights from the input layer to the hidden layer
     then output_neurons*hidden_neurons elements are connecting weights from the hidden layer to the output layer
     then hidden_neurons elements are the biases of the hidden layer neurons
     finally there are output_neurons biases for the output neurons

  """
  #unpacking the arguments:
  aux0=0
  aux1=aux0+input_neurons*hidden_neurons
  w0=X[aux0:aux1].reshape((hidden_neurons,input_neurons))
  aux0=aux1
  aux1=aux0+output_neurons*hidden_neurons
  w1=X[aux0:aux1].reshape((output_neurons,hidden_neurons))
  aux0=aux1
  aux1=aux0+hidden_neurons
  b0=X[aux0:aux1]
  aux0=aux1
  aux1=aux0+output_neurons
  b1=X[aux0:aux1]

  #all the neural network math is in these two lines:
  o0=1/(1+np.exp(-inputs.dot(w0.transpose())+b0))
  o1=1/(1+np.exp(    -o0.dot(w1.transpose())+b1))
  return o1

def cost_function(X):
  return ((targets-propagate(inputs,X))**2).sum()

def matrix_propagate(inputs,X):
  o1=np.array([propagate(inputs,X[i])for i in range(len(X))])
  return o1

def matrix_cost_function(X):
  return ((targets-matrix_propagate(inputs,X))**2).sum(axis=1).sum(axis=1)



#system state:
X=5*(np.random.random(N)-0.5)

bounds=np.zeros((N,2))
bounds[:,0]=-30
bounds[:,1]=30


#trying the optimum position using a conventional optimizer:
from scipy.optimize import minimize
res=minimize(cost_function ,X,
             method='L-BFGS-B',
             bounds=bounds.tolist(),
             options={'maxiter':500})

optimized_x_scipy=res.x


results=[]
repetitions=32
aux0=np.argmax(targets,axis=1)
'''
for i in range(repetitions):
  pso=bio.PSO(costfunction=matrix_cost_function,
            dimensions=N,
            lb=bounds[:,0],ub=bounds[:,1],
            maxiter=500,
            n=30,w0=0.9,wf=0.1,c1=2,c2=2,max_v=5,ini_v=5/10)
  y,x=pso.run()
  Y=propagate(inputs,x)
  aux1=np.argmax(Y,axis=1)
  success_rate=np.sum(aux0==aux1)/len(aux0)
  print("PSO success rate (%):",success_rate*100)
  results.append(['PSO',i,y,x,success_rate])

for i in range(repetitions):
  pso_obl=bio.PSO_OBL(costfunction=matrix_cost_function,
            dimensions=N,
            lb=bounds[:,0],ub=bounds[:,1],
            maxiter=500,
            n=30,w0=0.9,wf=0.1,c1=2,c2=2,max_v=5,ini_v=5/10)
  y,x=pso_obl.run()
  Y=propagate(inputs,x)
  aux1=np.argmax(Y,axis=1)
  success_rate=np.sum(aux0==aux1)/len(aux0)
  print("PSO_OBL success rate (%):",success_rate*100)
  results.append(['PSO_OBL',i,y,x,success_rate])

for i in range(repetitions):
  de=bio.DE(costfunction=matrix_cost_function,
            dimensions=N,
            lb=bounds[:,0],ub=bounds[:,1],
            maxiter=500,
            n=30,f=1.25,c=0.9)
  y,x=de.run()
  Y=propagate(inputs,x)
  aux1=np.argmax(Y,axis=1)
  success_rate=np.sum(aux0==aux1)/len(aux0)
  print("DE success rate (%):",success_rate*100)
  results.append(['DE',i,y,x,success_rate])

for i in range(repetitions):
  de_obl=bio.DE_OBL(costfunction=matrix_cost_function,
            dimensions=N,
            lb=bounds[:,0],ub=bounds[:,1],
            maxiter=500,
            n=30,f=1.25,c=0.9)
  y,x=de_obl.run()
  Y=propagate(inputs,x)
  aux1=np.argmax(Y,axis=1)
  success_rate=np.sum(aux0==aux1)/len(aux0)
  print("DE_OBL success rate (%):",success_rate*100)
  results.append(['DE_OBL',i,y,x,success_rate])

f=open("optimization_for_iris_raw_data.pickle",'wb')
pickle.dump(results,f)
f.close()
#'''

f=open("optimization_for_iris_raw_data.pickle",'rb')
results=pickle.load(f)
f.close()

print(r"""\begin{table*}[p]
 \centering
 \caption{Valores da função custo após 500 iterações e 32 repetições.}
 \begin{tabular}{lllccccrr}
  \toprule
  Algoritmo & & & média & mediana & mínimo & desvio padrão & classificação melhor que 95% (\%) & erro zero de classificação (\%) \\
  \midrule""")

for a in ['PSO','PSO_OBL','DE','DE_OBL']:
  r=np.array([[i[2],i[-1]] for i in results if i[0]==a])
  m=np.mean(r,axis=0)
  s=np.std(r,axis=0)
  md=np.median(r,axis=0)
  min_cost=np.min(r,axis=0)
  max_cost=np.max(r,axis=0)
  success_rate_95=np.sum(r[1,:]>=0.95)/repetitions
  success_rate_100=np.sum(r[1,:]>=0.9999)/repetitions
  print(a.replace('_',r'\_')+' & S=30 & N=27 & '+str(m[0])+' & '+str(md[0])+' & '+str(min_cost[0])+' & '+str(s[0])+' & '+str(success_rate_95*100)+' & '+str(success_rate_100*100)+r' \\')

print(r"""\bottomrule
 \end{tabular}
\end{table*}""")

from scipy.stats import ks_2samp
from scipy.stats import kstest
from scipy.stats import kruskal
from scipy.stats import ranksums

distributions={}

for a in ['PSO','PSO_OBL','DE','DE_OBL']:
  r=np.array([i[2] for i in results if i[0]==a])
  distributions[a]=r.copy()
  m=np.mean(r)
  s=np.std(r)
  print(a,kstest((r-m)/s,'norm'))
  print(r)
  md=np.median(r)
  min_cost=np.min(r)
  max_cost=np.max(r)

list(distributions.values())
a=[distributions[i] for i in ['PSO','PSO_OBL','DE','DE_OBL']]
print(kruskal(*a))
print(kruskal(a[0],a[1]))
print(kruskal(a[0],a[2]))
print(kruskal(a[0],a[3]))
print(kruskal(a[1],a[2]))
print(kruskal(a[1],a[3]))
print(kruskal(a[2],a[3]))

print('ranksums')
print(ranksums(a[0],a[1]))
print(ranksums(a[0],a[2]))
print(ranksums(a[0],a[3]))
print(ranksums(a[1],a[2]))
print(ranksums(a[1],a[3]))
print(ranksums(a[2],a[3]))
#n=np.array([i[2] for i in results]).argmin()
#x=results[n][3].copy()
#res=minimize(cost_function,x.copy(),method='L-BFGS-B',bounds=bounds.tolist(),options={'maxiter':500})
'''
print(results[n])
print()
print(cost_function(x.copy()))
print(matrix_cost_function(np.array([x.copy()])))
print()
print(res)
#'''
'''
normaliza entrada entre -1 e 1


#pso:
numero de neuronios=15 ou 4

30 particulas
500 iter
w=2. #inertia
wmax=0.9
wmin=0.1
c1=2
c2=2
dt=0.8 (ou 1.0)
'''