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

def ReconstructionImage(img, optX):
 delta = 254.0/(optX.size+1)
 intensityInterval = delta
 [Width, Height] = np.shape(img)
 img2 = np.copy(img)
 img[ img2 <= optX[1]] = intensityInterval
 for i in range(2, optX.size):
   intensityInterval +=delta
   img[np.logical_and((optX[i-1] < img2),(optX[i] >= img2))  ] = int(intensityInterval)
 intensityInterval +=delta
 img[ img2 > optX[-1]] =intensityInterval
 
