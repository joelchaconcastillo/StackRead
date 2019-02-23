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

def OptimizationDE(minv, maxv, Mt, AccumPi, AccumiPi, dimension):
  npop =  max(50, dimension)
  #Initialization....
  pop = np.random.randint( low = minv, high = maxv, size=(npop, dimension))
  evaluations = np.zeros(npop) 
  ##Elite...
  databest = np.zeros(dimension)
  bestFitness = -10000000
  #evaluation..
  for target in range(0, npop):

   #pop[target,:] = np.sort(pop[target,:])
   evaluations[target] = ObjectiveFunction(pop[target,:], minv, maxv, Mt, AccumPi, AccumiPi)
   if evaluations[target] > bestFitness:
     bestFitness = evaluations[target]
     databest = np.copy(pop[target,:])	

  for gen in range(0, 50):
     for target in range(0, npop):
       trial = np.copy(pop[target])
       #getting  two numbers...
       r1 = random.randint(0, npop-1) 
       while r1 == target:
        r1 = random.randint(0, npop-1) 
       r2 = random.randint(0, npop-1) 
       while r2 == target and r1 == r2:
        r2 = random.randint(0, npop-1) 
       ##Mutation and crossover...
       pr = random.uniform(0, 1)
       index = random.randint(0, dimension-1) 
#       F = 0.9
       CR = random.uniform(0, 1)
       for d in range(0, dimension):
	if random.uniform(0, 1) <= CR or index == d:
         #trial[d] =  pop[target][d] + F*( pop[r1][d] - pop[r2][d])
         diff = pop[r1][d] - pop[r2][d]
         trial[d] =  pop[target][d] + (np.sign(diff))*random.uniform(1, abs(diff) )
        else:
         trial[d] = pop[target][d]
        #print trial[d]
        trial[d] = min(maxv, trial[d])
        trial[d] = max(minv, trial[d])
       #trial = np.sort(trial)
       objtrial = ObjectiveFunction(trial, minv, maxv, Mt, AccumPi, AccumiPi)
       ##Selection
       if objtrial > evaluations[target]:
        pop[target] = np.copy(trial)
        evaluations[target] = objtrial 
	###record the best values 
       if evaluations[target] > bestFitness: 
        bestFitness = evaluations[target]
        databest = np.copy(pop[target,:])	
     #print bestFitness
  optX = np.zeros(dimension+1)
  optX[0] = bestFitness
  optX[1:dimension+1] = np.sort(databest)
  return optX

def GeneralizedOtsuDE(filename, Classes):

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
 optX = OptimizationDE( minv, maxv, Mt, AccumPi, AccumiPi, Classes-1)
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

