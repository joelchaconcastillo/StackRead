import math
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import random

def ObjectiveFunction(combinationThresholds, minv, maxv, Mt, AccumPi, AccumiPi):
  ###check repetitions....
  combinationThresholds = np.sort(combinationThresholds)
  k = combinationThresholds.size+1
  epsilon = 0.00000001
  Wi = np.zeros(k)
  Mu = np.zeros(k)
  i = int(combinationThresholds[0])
  Wi[0] = AccumPi[i-1]
  Mu[0] = (AccumiPi[i-1])/(Wi[0]+epsilon)# scaled[minv:i].dot(Pi[minv:i])/Wi[0]
  for z in range(1, k-1):
     previntensity = int(combinationThresholds[z-1])+1
     nextintensity = int(combinationThresholds[z])
     Wi[z] = AccumPi[nextintensity] - AccumPi[previntensity-1]# np.sum(Pi[previntensity:nextintensity])
     Mu[z] = (AccumiPi[nextintensity] - AccumiPi[previntensity-1])/(Wi[z]+epsilon)# scaled[previntensity:nextintensity].dot(Pi[previntensity:nextintensity])/(Wi[z])
  previntensity = int(combinationThresholds[-1])+1
  Wi[-1] =  AccumPi[maxv] - AccumPi[previntensity-1]#
  Mu[-1] = (AccumiPi[maxv] - AccumiPi[previntensity-1])/(Wi[-1]+epsilon) # scaled[previntensity:maxv].dot(Pi[previntensity:maxv])/Wi[-1]
  ##computing variance...
  return Wi.dot(np.power(Mu-Mt,2))

def Gradient(X, minv, maxv, Mt, AccumPi, AccumiPi):
    gradient =np.zeros(X.size)
    for d in range(0, X.size):
     Xi1 = np.copy(X)
     Xi2 = np.copy(X) 
     Xi1[d] +=1
     Xi2[d] -=1
     if Xi1[d] > maxv:
	Xi1[d]=minv
     if Xi2[d] < minv:
	Xi2[d]=maxv
     gradient[d] = (ObjectiveFunction(Xi1, minv, maxv, Mt, AccumPi, AccumiPi)-ObjectiveFunction(Xi2, minv, maxv, Mt, AccumPi, AccumiPi))
    return gradient

def GradientDescentMethod(minv, maxv, Mt, AccumPi, AccumiPi, dimension):
  delta = float(maxv-minv)/dimension
  X =  np.zeros(dimension)# np.random.uniform( low = minv, high = maxv, size=(dimension))
  T = 0
  for i in range(0,dimension):
    X[i] = T
    T+=delta
  X[ X > maxv] = maxv
  X[ X < minv] = minv
  alpha = 0.001 #stepsize 
  maxite = 500
  for ite in range(0, maxite):
    alpha =  np.random.uniform(0.4,0.9)
    Xtmp = np.copy(X)
    Xtmp += (alpha*Gradient(Xtmp, minv, maxv, Mt, AccumPi, AccumiPi))
    Xtmp[ Xtmp > maxv] = minv
    Xtmp[ Xtmp < minv] = maxv
  #  if ObjectiveFunction(Xtmp, minv, maxv, Mt, AccumPi, AccumiPi) > ObjectiveFunction(X, minv, maxv, Mt, AccumPi, AccumiPi):
    X = np.copy(Xtmp)
  optX = np.zeros(dimension+1)
  optX[0] = ObjectiveFunction(X, minv, maxv, Mt, AccumPi, AccumiPi)
  optX[1:dimension+1] = np.sort(X.astype(int))
  return optX

def GeneralizedOtsuGradient(filename, Classes):

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
 optX = GradientDescentMethod( minv, maxv, Mt, AccumPi, AccumiPi, Classes-1)
 print "Thresholds...."
 print optX
 delta = 254.0/(optX.size+1)
 intensityInterval = delta
 [Width, Height] = np.shape(img)
 img[ img < optX[1]] =intensityInterval
 for i in range(2, optX.size):
   intensityInterval +=delta
   img[np.logical_and((optX[i-1] > img),(optX[i] <= img))  ] = intensityInterval
 intensityInterval +=delta
 img[ img > optX[-1]] =intensityInterval
 return img

