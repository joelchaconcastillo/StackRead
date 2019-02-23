import math
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import random
def myKapurK(filename, Classes):
 img = misc.imread(filename, flatten=True, mode='I')
 img = img.astype(int)
 [Width, Height] = np.shape(img)
 return GeneralizedKapur(img, Classes)

def EvaluateThresholds(combinationThresholds,k, minv, maxv, Mt, optX, AccumPi, AccumiPi):
  TotalEntropy = 0 
  epsilon = 0.01
  i = int(combinationThresholds[0])
  PS = AccumPi[i-1]+epsilon
  SumLogPi = AccumiPi[i-1]
  TotalEntropy += math.log(PS) - SumLogPi/PS
  for z in range(1, k):
     previntensity = int(combinationThresholds[z-1])+1
     nextintensity = int(combinationThresholds[z])
     PS = AccumPi[nextintensity] - AccumPi[previntensity-1] + epsilon
     SumLogPi = (AccumiPi[nextintensity] - AccumiPi[previntensity-1])
     TotalEntropy += math.log(PS) - SumLogPi/PS

  previntensity = int(combinationThresholds[k-1])+1
  PS = AccumPi[maxv] - AccumPi[previntensity-1]+epsilon
  SumLogPi = (AccumiPi[maxv] - AccumiPi[previntensity-1])
  TotalEntropy += math.log(PS) - SumLogPi/PS
  if TotalEntropy >  optX[0]:
	optX[0] = TotalEntropy
	optX[1:optX.size] = combinationThresholds
##Recursive calling...
def Combination(combinationThresholds, size, k, index, setThresholds, i, minv, maxv, Mt, optX, AccumPi, AccumiPi  ):

 ##Current combination is ready
 if index == k:
   EvaluateThresholds(combinationThresholds, k, minv, maxv, Mt, optX, AccumPi, AccumiPi )
   return 0
 ##No more elemnts are there to put in combinationThresholds
 if i >= size:
   return 0

 combinationThresholds[index] = setThresholds[i]

 Combination(combinationThresholds, size, k, index+1, setThresholds, i+1, minv, maxv, Mt, optX, AccumPi, AccumiPi)
 while((setThresholds[i] == setThresholds[i+1])):
	i+=1
 Combination(combinationThresholds, size, k, index, setThresholds, i+1, minv, maxv, Mt, optX, AccumPi, AccumiPi)



def GeneralizedKapur(img, Classes):
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
  if Pi[i] > 0 :
    sumipi += Pi[i]*math.log(Pi[i])
  AccumPi[i] = sumpi
  AccumiPi[i] =  sumipi
 combinationThresholds = np.zeros(Classes-1)
 optX = np.zeros(combinationThresholds.size+1)
 optX[0] = -10000000
 Combination(combinationThresholds, setThresholds.size-1, Classes-1, 0, setThresholds, 0, minv, maxv, Mt, optX, AccumPi, AccumiPi)
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
