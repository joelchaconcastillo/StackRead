import math
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import random
from Global import ObjectiveFunction, ReconstructionImage

def OptimizationDE(npop, Niterations, minv, maxv, Mt, AccumPi, AccumiPi, dimension):
  #Initialization....
  pop = np.random.randint( low = minv, high = maxv, size=(npop, dimension))
  evaluations = np.zeros(npop) 
  ##Elite...
  databest = np.zeros(dimension)
  bestFitness = -10000000
  #evaluation..
  for target in range(0, npop):
   #print pop
   #pop[target,:] = np.sort(pop[target,:])
   evaluations[target] = ObjectiveFunction(pop[target,:], minv, maxv, Mt, AccumPi, AccumiPi)
   if evaluations[target] > bestFitness:
     bestFitness = evaluations[target]
     databest = np.copy(pop[target,:])	

  for gen in range(0, Niterations):
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
        if trial[d] > maxv:
	 trial[d] = minv+1
	if trial[d] < minv:
	 trial[d] = maxv-1
       objtrial = ObjectiveFunction(trial, minv, maxv, Mt, AccumPi, AccumiPi)
       ##Selection
       if objtrial > evaluations[target]:
        pop[target] = np.copy(trial)
        evaluations[target] = objtrial 
	###record the best values 
       if evaluations[target] > bestFitness: 
        bestFitness = evaluations[target]
        databest = np.copy(pop[target,:])	
  optX = np.zeros(dimension+1)
  optX[0] = bestFitness
  optX[1:dimension+1] = np.sort(databest)
  return optX

def GeneralizedOtsuDE(filename, Classes, PopulationSize, Niterations):
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
 optX = OptimizationDE(PopulationSize, Niterations, minv, maxv, Mt, AccumPi, AccumiPi, Classes-1)
 ReconstructionImage(img, optX)
 return img, optX[0], optX[1:optX.size]

