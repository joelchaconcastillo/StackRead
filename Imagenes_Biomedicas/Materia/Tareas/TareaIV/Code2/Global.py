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

def ObjectiveFunctionKapur(combinationThresholds, minv, maxv, Mt, AccumPi, AccumiPi):
  k = combinationThresholds.size
  TotalEntropy = 0 
  epsilon = 0.01
  i = int(combinationThresholds[0])
  PS = AccumPi[i-1]+epsilon
  PS = min(1.0-epsilon,PS)
  PS = max(epsilon, PS)
  SumLogPi = AccumiPi[i-1]
  TotalEntropy += math.log(PS) - SumLogPi/PS
  for z in range(1, k):
     previntensity = int(combinationThresholds[z-1])+1
     nextintensity = int(combinationThresholds[z])
     PS = AccumPi[nextintensity] - AccumPi[previntensity-1] + epsilon
     PS = min(1.0-epsilon,PS)
     PS = max(epsilon, PS)
     SumLogPi = (AccumiPi[nextintensity] - AccumiPi[previntensity-1])
     TotalEntropy += math.log(PS) - SumLogPi/PS

  previntensity = int(combinationThresholds[k-1])+1
  PS = AccumPi[maxv] - AccumPi[previntensity-1]+epsilon
  PS = min(1.0-epsilon,PS)
  PS = max(epsilon, PS)
  SumLogPi = (AccumiPi[maxv] - AccumiPi[previntensity-1])
  TotalEntropy += math.log(PS) - SumLogPi/PS
  return TotalEntropy

def ReconstructionImage(img, optX):
 [Width, Height] = np.shape(img)
 img2 = np.copy(img)
 img[ img2 <= optX[1]] = 0# intensityInterval
 for i in range(2, optX.size):
   img[np.logical_and((optX[i-1] < img2),(optX[i] >= img2))  ] = (optX[i]+optX[i-1])/2 #int(intensityInterval)
 img[ img2 > optX[-1]] = 254#intensityInterval
 
def PSNR(img, imgRef):
 [Width, Height] = np.shape(img)
 MSE =  np.sum(np.power(img - imgRef,2))/(Width*Height)+0.00001
 return 10*np.log( (255*255)/MSE)/np.log(10)
def SSIM(img, imgRef):
 MuI = np.mean(img.flatten()) 
 MuJ = np.mean(imgRef.flatten()) 
 SigmaI = np.std(img.flatten()) 
 SigmaJ = np.std(imgRef.flatten()) 
 SigmaIJ = np.corrcoef(img.flatten(), imgRef.flatten())[0,1]
 SigmaIJ = np.cov(img.flatten(), imgRef.flatten())[0,1]
 C1 = 6.5025
 C2 = 58.5225
 return ((2.0*MuI*MuJ+C1)*(2.0*SigmaIJ + C2))/(  ( MuI*MuI + MuJ*MuJ + C1 )*( SigmaI*SigmaI + SigmaJ*SigmaJ + C2 ))
  
