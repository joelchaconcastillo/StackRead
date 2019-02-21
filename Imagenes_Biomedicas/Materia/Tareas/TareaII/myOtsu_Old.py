import math
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
def myOtsuK(filename, Classes):
 img = misc.imread(filename, flatten=True, mode='I')
 img = img.astype(int)
 [Width, Height] = np.shape(img)
 if Classes==2:
  return Otsu2Optimized(img, Classes)
 if Classes > 2:
  return GeneralizedOtsu(img, Classes)

def GeneralizedOtsu(img, Classes):
 imgFlat = img.flatten()
 minv = np.min(img)
 maxv = np.max(img)
 Pi = np.zeros(maxv+1)
 Wi = np.zeros(Classes)
 Mu = np.zeros(Classes)
 Thresholds = np.zeros(Classes)
 for i in range(0, imgFlat.size):
  Pi[imgFlat[i]] +=1.0/imgFlat.size
 maxvariance = -10000000
 Threshold = 0
 scaled = np.arange(0, maxv)
 Mt = scaled.dot(Pi)
 for i in range(minv, maxv):
  Wi[0] = np.sum(Pi[minv:i])
  Mu[0] = scaled[minv:i].dot(Pi[minv:i])/Wi[0]
  Wi[1] = np.sum(Pi[(i+1):maxv])
  Mu[1] = scaled[(i+1):maxv].dot(Pi[(i+1):maxv])/Wi[1]
  ##computing variance...
  variance = Mu.dot(np.power(Mu-Mt,2))
  if variance > maxvariance:
     maxvariance = variance
     Threshold = i
 img[img < Threshold] = 0
 img[img >= Threshold] = 254
 return img

def Otsu2Optimized(img, Classes):
 [Width, Height] = np.shape(img)
 imgFlat = img.flatten()
 minv = np.min(img)
 maxv = np.max(img)
 Pi = np.zeros(maxv+1)
 Wi = np.zeros(Classes)
 Mu = np.zeros(Classes)
 Thresholds = np.zeros(Classes)
 for i in range(0, imgFlat.size):
  Pi[imgFlat[i]] +=1.0/imgFlat.size
  
 for i in range(minv, maxv):
  Wi[1] += Pi[i]
 for i in range(minv, maxv):
  Mu[1] += (i*Pi[i])/Wi[1]
 Mt = Mu[1]
 maxvariance = -10000000
 Threshold = 0
 for i in range(minv, maxv):
  Wi[0] += Pi[i]
  Mu[0] =  np.arange(minv, i).dot(Pi[minv:i])/Wi[0]
  Wi[1] -= Pi[i]
  Mu[1] =  np.arange(i+1, maxv).dot(Pi[i+1:maxv])/Wi[1]
  ##computing variance...
  variance = Wi.dot(np.power(Mu-Mt,2))
  if variance > maxvariance:
     maxvariance = variance
     Threshold = i
 print Threshold
 Threshold-=10
 img[img < Threshold] = 0
 img[img >= Threshold] = 254
 return img

