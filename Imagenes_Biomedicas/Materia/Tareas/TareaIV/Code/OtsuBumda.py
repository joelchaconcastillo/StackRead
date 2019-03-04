import math
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import random
from Global import ObjectiveFunction

def GeneralizedOtsuBumda(filename, Classes, PopulationSize, Niterations):
 img = misc.imread(filename, flatten=True, mode='I')
 img = img.astype(int)
 [Width, Height] = np.shape(img)
 imgFlat = img.flatten()
 minv = np.min(img)
 maxv = np.max(img)
 ##Generating factible solutions...
 setThresholds = np.arange(minv+1, maxv) 
 Pi = np.zeros(maxv+1)
 AccumPi = np.zeros(maxv+1)
 scaled = np.arange(0, maxv+1)
 AccumiPi = np.zeros(maxv+1)
 Mt = scaled.dot(Pi)
 for i in range(0, imgFlat.size):
  Pi[imgFlat[i]] +=(1.0/imgFlat.size)
 sumpi = 0.0
 sumipi = 0.0
 for i in range(minv, maxv+1):
  sumpi += Pi[i]
  sumipi += float(i)*Pi[i]
  AccumPi[i] = sumpi
  AccumiPi[i] =  sumipi

 combinationThresholds = np.zeros(Classes-1)
 optX = OptimizationBumda( 100, 100, Classes-1,minv, maxv, Mt, AccumPi, AccumiPi)
 #print "Thresholds...."
# print optX
 delta = 254.0/(optX.size+1)
 intensityInterval = delta
 [Width, Height] = np.shape(img)
 img2 = np.copy(img)
 img[ img2 <= optX[1]] =intensityInterval
 for i in range(2, optX.size):
   intensityInterval +=delta
   img[np.logical_and((optX[i-1] < img2),(optX[i] >= img2))  ] = int(intensityInterval)
 intensityInterval +=delta
 img[ img2 > optX[-1]] =intensityInterval
 return img

#def ObjectiveFunction(X):
#  return (X-10).dot(X-10)

def OptimizationBumda(Maxite, N, Dimension, minv, maxv, Mt, AccumPi, AccumiPi):
 #Initialization
 feval= np.zeros(N)
 Population = np.random.uniform(low=0, high=254, size=(N, Dimension))
 Elite = Population[0,:]
 for i in  range(0,N):
  for d in range(0,Dimension):
    Population[i,d] = max(Population[i,d], minv)
    Population[i,d] = min(Population[i,d], maxv)
  feval[i] =  -ObjectiveFunction(Population[i,:], minv, maxv, Mt, AccumPi, AccumiPi)
#ObjectiveFunction(Population[i,:]);
  Theta = np.amax(feval)
 Mu = np.zeros(Dimension)
 Sigma = np.zeros(Dimension)
 for i in range(0, Maxite):
    #Sorting
    IndexSorted = np.argsort(feval)
    Elite = np.copy(Population[IndexSorted[0],:])
    #Truncating method
    if i > 0:
     Theta = min( feval[IndexSorted[N/2]], np.amax(feval[feval <= Theta]))
    Gi =  np.amax(feval[feval<=Theta]) - feval[feval<=Theta] + 1
    for d in range(0,Dimension):
     #Computing Mean 
     Mu[d] = Population[feval<=Theta, d].dot(Gi)/np.sum(Gi)
     #Computing variance
     Sigma[d] = Gi.dot(np.power(Population[feval<=Theta, d ] -  Mu[d], 2)) / (1.0 + np.sum(Gi)) 
     #Generating
     Population[:,d] = np.random.normal(Mu[d], np.sqrt(Sigma[d]), N) 
     Population[0,:] = np.copy(Elite)
     ##Evaluation...
    for i in  range(0,N):
      for d in range(0,Dimension):
        Population[i,d] = max(Population[i,d], minv)
        Population[i,d] = min(Population[i,d], maxv)
      feval[i] =  -ObjectiveFunction(Population[i,:], minv, maxv, Mt, AccumPi, AccumiPi)
 optX = np.zeros(Dimension+1)
 optX[0] = feval[0]
 optX[1:Dimension+1] = np.sort(Elite)
 return optX
