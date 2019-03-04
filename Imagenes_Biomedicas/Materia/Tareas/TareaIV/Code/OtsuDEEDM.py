import math
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import random
from Global import ObjectiveFunction

##this function combines both populations and select the best...
def replacement(Ptarget, evaluationsTarget, Ptrial, evaluationsTrial, minimumDistance):
 [n, m] = np.shape(Ptarget)
  candidates = np.zeros((2*n, m))
  fcandidates = np.zeros(2*n) 
  survivors= np.zeros((2*n, m))
  fsurvivors= np.zeros(2*n) 
  selected = np.zeros(2*n)
  penalized = []
  fpenalized = []

  ##joining...
  for i in range(0,n):
     candidates[i] = np.copy(Ptarget[i])
     fcandidates[i] = evaluationsTarget[i]
  for i in range(0,n):
     candidates[i+n] = np.copy(Ptrial[i])
     fcandidates[i+n] = evaluationsTrial[i]


  ##preprocessing .....
  distances = np.ones(2*n)*100000000000000
  ##sorting based in the fitness....
  IndexSorted = np.argsort(fcandidates)
  int index = 0 
  survivors[index] = np.copy.candidates[IndexSorted[0]]
  fsurvivors[index] = fcandidates[IndexSorted[0]]
  selected[IndexSorted[0]] = 1
  index +=1

  for i in range(1, 2*n):
   if selected[index[i]] == 1:
    continue
   for j in range(i+1, 2*n):
     if selected[index[j]] == 1:
       continue
     if np.linealg.norm( survivors[j] - candidates[i]) < minimumDistance:
       penalized.vstack(candidates[i])
       fpenalized.
     
       continue
     selected[index] = candidates[i]
     fsurvivors[index] = candidates[i]
     index += 1 




  for i in range(0, 2*n):
    for j in range(0, 2*n):
      distances[i] = min(distances[i], np.linealg.norm(candidates[i,:] - candidates[j,:]))


  

def OptimizationDEEDM(npop, Niterations, minv, maxv, Mt, AccumPi, AccumiPi, dimension):
  #Initialization....
  Ptarget = np.random.randint( low = minv, high = maxv, size=(npop, dimension))
  Ptrial = np.random.randint( low = minv, high = maxv, size=(npop, dimension))
  evaluationsTarget = np.zeros(npop) 
  evaluationsTrial = np.zeros(npop) 
  ##Elite...
  databest = np.zeros(dimension)
  bestFitness = -10000000
  #evaluation..
  for target in range(0, npop):
   evaluationsTarget[target] = ObjectiveFunction(Ptarget[target,:], minv, maxv, Mt, AccumPi, AccumiPi)
   if evaluationsTarget[target] > bestFitness:
     bestFitness = evaluationsTarget[target]
     databest = np.copy(Ptarget[target,:])	

  for gen in range(0, Niterations):
     for target in range(0, npop):
       Ptrial[target] = np.copy(Ptarget[target])
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
         diff = Ptarget[r1][d] - Ptarget[r2][d]
         Ptrial[d] =  Ptarget[target][d] + (np.sign(diff))*random.uniform(1, abs(diff) )
        else:
         Ptrial[d] = Ptarget[target][d]
        if Ptrial[d] > maxv:
	 Ptrial[d] = minv+1
	if Ptrial[d] < minv:
	 Ptrial[d] = maxv-1
       #trial = np.sort(trial)
       evaluationsTrial[target] = ObjectiveFunction(Ptrial[target,:], minv, maxv, Mt, AccumPi, AccumiPi)
       ##Selection
       if evaluationsTrial[target] > evaluationsTarget[target]:
        Ptarget[target] = np.copy(Ptrial[target])
        evaluationsTarget[target] = evaluationsTrial[target]
	###record the best values 
       if evaluationsTarget[target] > bestFitness: 
        bestFitness = evaluationsTarget[target]
        databest = np.copy(Ptarget[target,:])	
       ########replacement phase...
     replacement(Ptarget, evaluationsTarget, Ptrial, evaluationsTrial, minimumDistance) 
#     print bestFitness
  optX = np.zeros(dimension+1)
  optX[0] = bestFitness
  optX[1:dimension+1] = np.sort(databest)
  return optX

def GeneralizedOtsuDEEDM(filename, Classes, PopulationSize, Niterations):
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
 optX = OptimizationDEEDM( PopulationSize, Niterations, minv, maxv, Mt, AccumPi, AccumiPi, Classes-1)
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

