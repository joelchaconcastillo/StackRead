import math
import os
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from myOtsu import myOtsuK
from OtsuDE import GeneralizedOtsuDE
#from OtsuDEEDM import GeneralizedOtsuDEEDM
from OtsuBumda import GeneralizedOtsuBumda
from OtsuGradient import GeneralizedOtsuGradient
from myKapur import myKapurK
from Landscapes import LandScapeOtsu, LandScapeKapur
#from creationReference import generating 

from skimage import data
from skimage import filters
from skimage import exposure

#################MULTI-LEVEL-THRESHOLD############################

f = open('files', "r")
#in each coordinate...
Clases=6
PopulationSize = 50
NIterations = 300
for x in f:
 #fig = plt.figure()
 #filename = 'BD1/'+x.rstrip()+'.jpg'
# imagen2 = myOtsuK(filename, Clases)
# imagenref = misc.imread('BD1/'+x.rstrip()+'.jpg', flatten=True, mode='I')
# val = filters.threshold_otsu(imagenref)
# print val
 #exit(0)
 #plt.imsave(os.path.splitext(filename)[0]+'_Otsu_k'+str(Clases)+'.eps', imagen2, cmap='gray')
 #plt.close()
##
# fig = plt.figure()
# filename = 'BD1/'+x.rstrip()+'.jpg'
# imagen2 = Fuzzy_Entropy('BD1/'+x.rstrip()+'.jpg')
# plt.imsave(os.path.splitext(filename)[0]+'_fuzzy_k'+str(Clases)+'.eps', imagen2, cmap='gray')
# plt.close()



 #fig = plt.figure()
 #filename = 'BD1/'+x.rstrip()+'.jpg'
 #imagen2 = myKapurK('BD1/'+x.rstrip()+'.jpg', Clases)
 #plt.imsave(os.path.splitext(filename)[0]+'_kapur_k'+str(Clases)+'.eps', imagen2, cmap='gray')
 #plt.close()

 fig = plt.figure()
 filename = 'images/'+x.rstrip()
 imagen2 = GeneralizedOtsuDE('images/'+x.rstrip(), Clases, PopulationSize, NIterations)
 plt.imsave(os.path.splitext(filename)[0]+'_OtsuDE_k'+str(Clases)+'.eps', imagen2, cmap='gray')
 
 plt.close()

 ##fig = plt.figure()
 ##filename = 'BD1/'+x.rstrip()+'.jpg'
 ##imagen2 = GeneralizedOtsuGradient('BD1/'+x.rstrip()+'.jpg', Clases)
 ##plt.imsave(os.path.splitext(filename)[0]+'_OtsuGradient_k'+str(Clases)+'.eps', imagen2, cmap='gray')
 ##plt.close()

 #fig = plt.figure()
 #filename = 'images/'+x.rstrip()
 #imagen2 = GeneralizedOtsuBumda('images/'+x.rstrip(), Clases, PopulationSize, NIterations)
 #plt.imsave(os.path.splitext(filename)[0]+'_OtsuBumda_k'+str(Clases)+'.eps', imagen2, cmap='gray')
 #plt.close()

# fig = plt.figure()
# filename = 'images/'+x.rstrip()
# imagen2 = GeneralizedOtsuDEEDM('images/'+x.rstrip(), Clases, PopulationSize, NIterations)
# plt.imsave(os.path.splitext(filename)[0]+'_OtsuDEEM_k'+str(Clases)+'.eps', imagen2, cmap='gray')
# plt.close()



 exit(0)


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
#in each coordinate...
for x in f:
 LandScapeOtsu('images/'+x.rstrip(), 2)
 LandScapeKapur('images/'+x.rstrip(), 2)

