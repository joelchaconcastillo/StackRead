import math
import os
import random
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import sys
from myOtsu import myOtsuK
from OtsuDE import GeneralizedOtsuDE
from OtsuDEEDM import GeneralizedOtsuDEEDM
from OtsuBumda import GeneralizedOtsuBumda
from OtsuGradient import GeneralizedOtsuGradient
from DE_Expectation import GeneralizedDEExpectation
from myKapur import myKapurK
from Landscapes import LandScapeOtsu, LandScapeKapur
from Global import PSNR, SSIM
from KapurDE import GeneralizedKapurDE
#from creationReference import generating 

#from skimage import data
#from skimage import filters
#from skimage import exposure

#################MULTI-LEVEL-THRESHOLD############################
random.seed(9001)
f = open('files', "r")
#in each coordinate...
Clases=3
PopulationSize = 30
NIterations = ((Clases-1)*10000)/PopulationSize

Reptetitions = 1
NAlgorithms = 6
NImages = 10
AveragePSNR = np.zeros((NAlgorithms,NImages))
AverageObjective = np.zeros((NAlgorithms,NImages))
AverageSSIM = np.zeros((NAlgorithms,NImages))
cont = 0
for x in f:
 dataPSNR = np.zeros((NAlgorithms, Reptetitions))
 dataSSIM = np.zeros((NAlgorithms, Reptetitions))
 dataobj = np.zeros((NAlgorithms, Reptetitions))
 for rep in range(0,Reptetitions):
  sys.stderr.write(str(rep)+"\n")
  filename = 'images/'+x.rstrip()
  imagenref = misc.imread('images/'+x.rstrip(), flatten=True, mode='I')
 # #fig = plt.figure()
  imagen2 = myOtsuK(filename, Clases)
  plt.imsave(os.path.splitext(filename)[0]+'_Otsu_k'+str(Clases)+'.eps', imagen2, cmap='gray')
 
 ##
 
  #fig = plt.figure()
  imagen2 = myKapurK('images/'+x.rstrip(), Clases)
  plt.imsave(os.path.splitext(filename)[0]+'_kapur_k'+str(Clases)+'.eps', imagen2, cmap='gray')
  #plt.close()
 
#  filename = 'images/'+x.rstrip()
  imagen2, obj, thresholds = GeneralizedOtsuDE('images/'+x.rstrip(), Clases, PopulationSize, NIterations)
  plt.imsave(os.path.splitext(filename)[0]+'_OtsuDE_k'+str(Clases)+'.eps', imagen2, cmap='gray')

#  dataPSNR[0, rep] = PSNR(imagen2, imagenref)
#  dataSSIM[0, rep] = SSIM(imagen2, imagenref)
#  dataobj[0, rep] = obj
# 
#  #fig = plt.figure()
  imagen2, obj, thresholds = GeneralizedOtsuGradient('images/'+x.rstrip(), Clases, NIterations)
  plt.imsave(os.path.splitext(filename)[0]+'_OtsuGradient_k'+str(Clases)+'.eps', imagen2, cmap='gray')
#  dataPSNR[1, rep] = PSNR(imagen2, imagenref)
#  dataSSIM[1, rep] = SSIM(imagen2, imagenref)
#  dataobj[1, rep] = obj
#
#  imagen2, obj, thresholds = GeneralizedOtsuBumda('images/'+x.rstrip(), Clases, PopulationSize, NIterations)
#  #plt.imsave(os.path.splitext(filename)[0]+'_OtsuBumda_k'+str(Clases)+'.eps', imagen2, cmap='gray')
#  dataPSNR[2, rep] = PSNR(imagen2, imagenref)
#  dataSSIM[2, rep] = SSIM(imagen2, imagenref)
#  dataobj[2, rep] = obj
# 
#  imagen2, obj, thresholds = GeneralizedOtsuDEEDM('images/'+x.rstrip(), Clases, PopulationSize, NIterations)
#  #plt.imsave(os.path.splitext(filename)[0]+'_OtsuDEEM_k'+str(Clases)+'.eps', imagen2, cmap='gray')
#  dataPSNR[3, rep] = PSNR(imagen2, imagenref)
#  dataSSIM[3, rep] = SSIM(imagen2, imagenref)
#  dataobj[3, rep] = obj

  imagen2, obj, thresholds = GeneralizedDEExpectation('images/'+x.rstrip(), Clases, PopulationSize, NIterations)
  dataPSNR[4, rep] = PSNR(imagen2, imagenref)
  dataSSIM[4, rep] = SSIM(imagen2, imagenref)
  dataobj[4, rep] = obj

  imagen2, obj, thresholds = GeneralizedKapurDE('images/'+x.rstrip(), Clases, PopulationSize, NIterations)
  dataPSNR[5, rep] = PSNR(imagen2, imagenref)
  dataSSIM[5, rep] = SSIM(imagen2, imagenref)
  dataobj[5, rep] = obj
 for k in range(0, NAlgorithms):
   AveragePSNR[k, cont] = np.average(dataPSNR[k,:])
   AverageObjective[k, cont] = np.average(dataobj[k,:])
   AverageSSIM[k, cont] = np.average(dataSSIM[k,:])
 cont +=1
 break
 print str(np.average(dataPSNR[0,:])) + " " +str(np.average(dataPSNR[1,:]))+" "+str(np.average(dataPSNR[2,:])) + " " +str(np.average(dataPSNR[3,:]))+ " " +str(np.average(dataPSNR[4,:]))+ " " +str(np.average(dataPSNR[5,:]))
 print str(np.average(dataSSIM[0,:])) + " " +str(np.average(dataSSIM[1,:]))+" "+str(np.average(dataSSIM[2,:])) + " " +str(np.average(dataSSIM[3,:]))+ " " +str(np.average(dataSSIM[4,:]))+ " " +str(np.average(dataSSIM[5,:]))
 print str(np.average(dataobj[0,:])) + " " +str(np.average(dataobj[1,:]))+" "+str(np.average(dataobj[2,:])) + " " +str(np.average(dataobj[3,:]))+ " " +str(np.average(dataobj[4,:]))+ " " +str(np.average(dataobj[5,:]))
 
print str(np.average(AveragePSNR[0,:])) + " " +str(np.average(AveragePSNR[1,:]))+" "+str(np.average(AveragePSNR[2,:])) + " " +str(np.average(AveragePSNR[3,:]))+ " " +str(np.average(AveragePSNR[4,:]))+ " " +str(np.average(AveragePSNR[5,:]))
print str(np.average(AverageSSIM[0,:])) + " " +str(np.average(AverageSSIM[1,:]))+" "+str(np.average(AverageSSIM[2,:])) + " " +str(np.average(AverageSSIM[3,:]))+ " " +str(np.average(AverageSSIM[4,:]))+ " " +str(np.average(AverageSSIM[5,:]))
print str(np.average(AverageObjective[0,:])) + " " +str(np.average(AverageObjective[1,:]))+" "+str(np.average(AverageObjective[2,:])) + " " +str(np.average(AverageObjective[3,:]))+ " " +str(np.average(AverageObjective[4,:]))+ " " +str(np.average(AverageObjective[5,:]))

 #imagenref = misc.imread('BD1/'+x.rstrip()+'_Reference.jpg', flatten=True, mode='I')
 #imagenref = misc.imread('BD1/'+x.rstrip()+'.jpg', flatten=True, mode='I')
 #imagenref = imagenref.astype(int)
 #plt.imshow(imagen2, cmap='gray')
# fig, (uno, dos) = plt.subplots(1,2)
# uno.imshow(imagenref, cmap='gray')
# dos.imshow(imagen2, cmap='gray');
# plt.show()


####################PLOTTING LANDSCAPE...###########################
f = open('files', "r")
##in each coordinate...
for x in f:
 LandScapeOtsu('images/'+x.rstrip(), 2)
 LandScapeKapur('images/'+x.rstrip(), 2)
 break
#
