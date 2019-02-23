import math
import os
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from myOtsu import myOtsuK
from Image_Thresholding_By_Minimizing_The_Measures_Of_Fuzziness import Fuzzy_Entropy
from OtsuDE import GeneralizedOtsuDE
from OtsuGradient import GeneralizedOtsuGradient
from myKapur import myKapurK
from Landscapes import LandScapeOtsu, LandScapeKapur

from skimage import data
from skimage import filters
from skimage import exposure

def Sensitivity(TP, FN):
   return  float(TP)/(TP+FN)
def Specifity(TN, FP):
   return float(TN)/(TN+FP)
def Accuracy(TP, TN, FP,FN):
   return float(TP+TN)/(TP + FP + TN + FN)
def MTCC(TP, TN, FP, FN):
   return  float((TP*TN)-(FP*FN))/math.sqrt( np.float64 ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
def JACCARD_INDEX(TP, FP, FN):
   return float(TP)/(FP+FN+TP)

#############################################
###Checking the performance considering all the images...
###TP = 0.0
###FP = 0.0
###TN = 0.0
###FN = 0.0
###f = open('files', "r")
####in each coordinate...
###for x in f:
### #imagen2 = Fuzzy_Entropy('BD1/'+x.rstrip()+'.jpg')
### #imagen2 = myOtsuK('BD1/'+x.rstrip()+'.jpg', 3)
#### imagen2 = myKapurK('BD1/'+x.rstrip()+'.jpg', 4)
#### imagen2 = GeneralizedOtsuDE('BD1/'+x.rstrip()+'.jpg', 3)
#### imagen2 = GeneralizedOtsuGradient('BD1/'+x.rstrip()+'.jpg', 3)
### #imagenref = misc.imread('BD1/'+x.rstrip()+'_Reference.jpg', flatten=True, mode='I')
### imagenref = misc.imread('BD1/'+x.rstrip()+'.jpg', flatten=True, mode='I')
### imagenref = imagenref.astype(int)
###
### LandScapeOtsu('BD1/'+x.rstrip()+'.jpg', 2)
### LandScapeKapur('BD1/'+x.rstrip()+'.jpg', 2)
#### fig, (uno, dos) = plt.subplots(1,2)
#### uno.imshow(imagenref, cmap='gray')
#### dos.imshow(imagen2, cmap='gray');
#### plt.show()
######
####### print imagen2
###### [Width, Height] = np.shape(imagenref)
###### TP1 =  np.sum( (imagenref.flatten() == 0) & (imagen2.flatten() == 254)) 
###### FP1 =  np.sum( (imagenref.flatten() == 254) & (imagen2.flatten()== 254)) 
###### TN1 =  np.sum( (imagenref.flatten() == 254) & (imagen2.flatten() == 0)) 
###### FN1 =  np.sum( (imagenref.flatten() == 0) & (imagen2.flatten() == 0)) 
###### TP = TP + TP1
###### FP = FP + FP1
###### TN = TN + TN1
###### FN = FN + FN1
###### print x
###### print str(Sensitivity(TP1, FN1)) + ' ' + str(Specifity(TN1, FP1)) + ' ' + str(Accuracy(TP1, TN1, FP1, FN1)) + ' ' +str(JACCARD_INDEX(TP1, FP1, FN1))
###
####print str(Sensitivity(TP, FN)) + ' ' + str(Specifity(TN, FP)) + ' ' + str(Accuracy(TP, TN, FP, FN))+  ' ' +str(JACCARD_INDEX(TP, FP, FN))

#################MULTI-LEVEL-THRESHOLD############################

f = open('files', "r")
#in each coordinate...
for Clases in range(2,3):
 for x in f:
 
  fig = plt.figure()
  filename = 'BD1/'+x.rstrip()+'.jpg'
  imagen2 = myOtsuK(filename, Clases)
 # imagenref = misc.imread('BD1/'+x.rstrip()+'.jpg', flatten=True, mode='I')
 # val = filters.threshold_otsu(imagenref)
 # print val
  #exit(0)
  plt.imsave(os.path.splitext(filename)[0]+'_Otsu_k'+str(Clases)+'.eps', imagen2, cmap='gray')
  plt.close()
 
 
  fig = plt.figure()
  filename = 'BD1/'+x.rstrip()+'.jpg'
  imagen2 = myKapurK('BD1/'+x.rstrip()+'.jpg', Clases)
  plt.imsave(os.path.splitext(filename)[0]+'_kapur_k'+str(Clases)+'.eps', imagen2, cmap='gray')
  plt.close()
 
  fig = plt.figure()
  filename = 'BD1/'+x.rstrip()+'.jpg'
  imagen2 = GeneralizedOtsuDE('BD1/'+x.rstrip()+'.jpg', Clases)
  plt.imsave(os.path.splitext(filename)[0]+'_OtsuDE_k'+str(Clases)+'.eps', imagen2, cmap='gray')
  plt.close()
 
  fig = plt.figure()
  filename = 'BD1/'+x.rstrip()+'.jpg'
  imagen2 = GeneralizedOtsuGradient('BD1/'+x.rstrip()+'.jpg', Clases)
  plt.imsave(os.path.splitext(filename)[0]+'_OtsuGradient_k'+str(Clases)+'.eps', imagen2, cmap='gray')
  plt.close()



 #imagenref = misc.imread('BD1/'+x.rstrip()+'_Reference.jpg', flatten=True, mode='I')
 #imagenref = misc.imread('BD1/'+x.rstrip()+'.jpg', flatten=True, mode='I')
 #imagenref = imagenref.astype(int)
 #plt.imshow(imagen2, cmap='gray')
# fig, (uno, dos) = plt.subplots(1,2)
# uno.imshow(imagenref, cmap='gray')
# dos.imshow(imagen2, cmap='gray');
# plt.show()


##################PLOTTING LANDSCAPE...###########################
f = open('files', "r")
#in each coordinate...
for x in f:
 LandScapeOtsu('BD1/'+x.rstrip()+'.jpg', 2)
 LandScapeKapur('BD1/'+x.rstrip()+'.jpg', 2)

