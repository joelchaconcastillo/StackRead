import math
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from mpl_toolkits.mplot3d import Axes3D

def ObjectiveFunctionOtsu(combinationThresholds, minv, maxv, Mt, AccumPi, AccumiPi):
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
  combinationThresholds = np.sort(combinationThresholds)
  k = combinationThresholds.size
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
  return TotalEntropy

def LandScapeOtsu(filename, Classes):
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

 x =[]#np.zeros(maxv-minv+1)
 y =[]#np.zeros(maxv-minv+1)
 z =[]#np.zeros(maxv-minv+1)
###checking Landscape...
 for i in range(minv, maxv, 5):
   for j in range(minv, maxv, 5):
    x = np.append(x, i)
    y = np.append(y, j)
    z = np.append(z, ObjectiveFunctionOtsu([i,j], minv, maxv, Mt, AccumPi, AccumiPi))
 fig = plt.figure()
 ax = fig.gca(projection='3d')
 ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
 plt.savefig(os.path.splitext(filename)[0]+'landscape_Otsu.eps')
 #plt.show()
 print "printing...."


def LandScapeKapur(filename, Classes):
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
  if Pi[i] > 0 :
    sumipi += Pi[i]*math.log(Pi[i])
  AccumPi[i] = sumpi
  AccumiPi[i] =  sumipi
 x =[]#np.zeros(maxv-minv+1)
 y =[]#np.zeros(maxv-minv+1)
 z =[]#np.zeros(maxv-minv+1)
###checking Landscape...
 for i in range(minv, maxv, 5):
   for j in range(minv, maxv, 5):
    x = np.append(x, i)
    y = np.append(y, j)
    z = np.append(z, ObjectiveFunctionKapur([i,j], minv, maxv, Mt, AccumPi, AccumiPi))
 fig = plt.figure()
 ax = fig.gca(projection='3d')
 ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
 plt.savefig(os.path.splitext(filename)[0]+'landscape_Kapur.eps')
# plt.show()
 print "printing...."



