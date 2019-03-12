import math
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import random
#from skimage import data
#from skimage import filters
#from skimage import exposure
from Global import ObjectiveFunction, ReconstructionImage, PSNR

def myOtsuKPSNR(filename, Classes):
 img = misc.imread(filename, flatten=True, mode='I')
 img = img.astype(int)
 [Width, Height] = np.shape(img)
# if Classes==2:
#  return Otsu2Optimized(img, Classes)
# if Classes > 2:
 return GeneralizedOtsuPSNR(img, Classes, filename)

def EvaluateThresholds(combinationThresholds,k, minv, maxv, Mt, optX, AccumPi, AccumiPi, filename):
#  epsilon=0.0000001
#  Wi = np.zeros(k+1)
#  Mu = np.zeros(k+1)
#  i = int(combinationThresholds[0])
#  Wi[0] = AccumPi[i-1]
#  Mu[0] = (AccumiPi[i-1])/(Wi[0]+epsilon)# scaled[minv:i].dot(Pi[minv:i])/Wi[0]
#  
#  for z in range(1, k):
#     previntensity = int(combinationThresholds[z-1])+1
#     nextintensity = int(combinationThresholds[z])
#     Wi[z] = AccumPi[nextintensity] - AccumPi[previntensity-1]# np.sum(Pi[previntensity:nextintensity])
#     Mu[z] = (AccumiPi[nextintensity] - AccumiPi[previntensity-1])/(Wi[z]+epsilon)# scaled[previntensity:nextintensity].dot(Pi[previntensity:nextintensity])/(Wi[z])
#
#  previntensity = int(combinationThresholds[k-1])+1
#  Wi[-1] =  AccumPi[maxv] - AccumPi[previntensity-1]#
#  Mu[-1] = (AccumiPi[maxv] - AccumiPi[previntensity-1])/(Wi[-1]+epsilon) # scaled[previntensity:maxv].dot(Pi[previntensity:maxv])/Wi[-1]
  img = misc.imread(filename, flatten=True, mode='I')
  img = img.astype(int)
  imgref = misc.imread(filename, flatten=True, mode='I')
  imgref = img.astype(int)
  tmp = np.zeros(combinationThresholds.size+1)
  tmp[1:tmp.size] = np.copy(combinationThresholds)
  ReconstructionImage(img, tmp)
  
#  z = np.append(z, PSNR(img, imgref))#np.append(z, ObjectiveFunctionKapur([i,j], minv, maxv, Mt, AccumPi, AccumiPi))

  ##computing variance...
#  variance = Wi.dot(np.power(Mu-Mt,2))
  variance = PSNR(img, imgref)
  if variance >  optX[0]:
	optX[0] = variance
	optX[1:optX.size] = combinationThresholds

##Recursive calling...
def Combination(combinationThresholds, size, k, index, setThresholds, i, minv, maxv, Mt, optX, AccumPi, AccumiPi, filename  ):

 ##Current combination is ready
 if index == k:
   EvaluateThresholds(combinationThresholds, k, minv, maxv, Mt, optX, AccumPi, AccumiPi, filename )
   return 0
 ##No more elemnts are there to put in combinationThresholds
 if i >= size:
   return 0

 combinationThresholds[index] = setThresholds[i]

 Combination(combinationThresholds, size, k, index+1, setThresholds, i+1, minv, maxv, Mt, optX, AccumPi, AccumiPi, filename)
 while((setThresholds[i] == setThresholds[i+1])):
	i+=1
 Combination(combinationThresholds, size, k, index, setThresholds, i+1, minv, maxv, Mt, optX, AccumPi, AccumiPi, filename)


def GeneralizedOtsuPSNR(img, Classes, filename):
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
 optX = np.zeros(combinationThresholds.size+1)
 optX[0] = -10000000
 Combination(combinationThresholds, setThresholds.size-1, Classes-1, 0, setThresholds, 0, minv, maxv, Mt, optX, AccumPi, AccumiPi, filename)
 print optX
 ReconstructionImage(img, optX)
 return img, optX[0], optX[1:optX.size]
