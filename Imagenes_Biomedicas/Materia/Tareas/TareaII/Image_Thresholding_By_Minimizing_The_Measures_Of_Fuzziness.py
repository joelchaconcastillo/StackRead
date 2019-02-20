import math
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from myOtsu import myOtsuK

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

def computingEntropy(Membershipvalue, H, minv, maxv):
   Entropy = 0
   for g in range(minv, maxv):
     SMa=0
     if Membershipvalue[g] > 0 and Membershipvalue[g] < 1:
      SMa = -Membershipvalue[g]*math.log(Membershipvalue[g]) - ( (1.0-Membershipvalue[g])*math.log(1.0-Membershipvalue[g]))
     Entropy += SMa*H[g]
   return Entropy

#computing of the measure fuziness
def measure_fuziness(X, mu0, mu1, threshold, H):
  maxv = np.max(X)
  minv = np.min(X)
  C = maxv - minv
  #[Width, Height] = np.shape(imagen)
  ##two classes..
  Membershipvalue = np.zeros(maxv+1)
  for i in range(minv, maxv):
       kindmean = 0
       if i <= threshold:
	  kindmean = mu0
       else:
	  kindmean = mu1
       Membershipvalue[i] = 1.0/(1.0 + (abs(i-kindmean)/C))
  return computingEntropy(Membershipvalue, H, minv, maxv)

def Fuzzy_Entropy(filename):
  #Obtaining the image in gray level...
  #imagen = misc.imread('BD1/Im001_1.jpg', mode='I')
  #imagen = misc.imread('BD1/Im001_1.jpg', flatten=True, mode='I')
  imagen = misc.imread(filename, flatten=True, mode='I')
  imagen2 = misc.imread(filename, flatten=True, mode='I')
  #imagen2 = misc.imread('BD1/Im001_1.jpg', flatten=True, mode='I')
  #imagen = Image.open('BD1/Im001_1.jpg')
  #imagen = imagen.convert('L')
  imagen = imagen.astype(int)
  [Width, Height] = np.shape(imagen)
  #print imagen
  #print np.max(imagen)
  #print np.min(imagen)
  H = np.zeros(np.max(imagen)+1)
  GH = np.zeros(np.max(imagen)+1)
  #print(imagen)
  ##computing a counter  of gray levels...
  Totalsum = 0
  Totalsumvalues = 0
  #print imagen[0]
  for i in range(0, Width-1):
    for j in range(0, Height-1):
  #    print  imagen[i][j]
      H[imagen[i][j]] += 1
      GH[imagen[i][j]] += imagen[i][j]
      Totalsum += 1
      Totalsumvalues += imagen[i][j]
  
  
  ##main procedure..
  maxv = np.max(imagen)
  minv = np.min(imagen)
  St = H[minv] 
  Stc = Totalsum
  Wt = GH[minv]
  Wtc = Totalsumvalues
  mu0 = 0
  mu1 = 0 
  opt_threshold = 0
  minMeasurementFuzziness = 1000000
  for t in range(minv, maxv):
     St += H[t]
     Stc -=  St
     Wt += GH[t]
     Wtc -= Wt
     mu0 = math.floor(Wt/St)    
     mu1 = math.floor(Wtc/Stc)    
     msfu = measure_fuziness(imagen, mu0, mu1, t, H)
     if msfu < minMeasurementFuzziness:
        minMeasurementFuzziness = msfu
        opt_threshold = t
  imagen2[:][:]=0
  for i in range(0, Width-1):
    for j in range(0, Height-1):
     if imagen[i][j] > opt_threshold:
      imagen2[i][j] = 254
  return imagen2

#############################################
###Checking the performance considering all the images...
TP = 0.0
FP = 0.0
TN = 0.0
FN = 0.0
f = open('files', "r")
#in each coordinate...
for x in f:
 #imagen2 = Fuzzy_Entropy('BD1/'+x.rstrip()+'.jpg')
 imagen2 = myOtsuK('BD1/'+x.rstrip()+'.jpg', 2)
 imagenref = misc.imread('BD1/'+x.rstrip()+'_Reference.jpg', flatten=True, mode='I')
 imagenref = imagenref.astype(int)

 fig, (uno, dos) = plt.subplots(1,2)
 uno.imshow(imagenref, cmap='gray')
 dos.imshow(imagen2, cmap='gray');
 plt.show()

# print imagen2
 [Width, Height] = np.shape(imagenref)
 TP1 =  np.sum( (imagenref.flatten() == 0) & (imagen2.flatten() == 254)) 
 FP1 =  np.sum( (imagenref.flatten() == 254) & (imagen2.flatten()== 254)) 
 TN1 =  np.sum( (imagenref.flatten() == 254) & (imagen2.flatten() == 0)) 
 FN1 =  np.sum( (imagenref.flatten() == 0) & (imagen2.flatten() == 0)) 
 TP = TP + TP1
 FP = FP + FP1
 TN = TN + TN1
 FN = FN + FN1
 print x
 print str(Sensitivity(TP1, FN1)) + ' ' + str(Specifity(TN1, FP1)) + ' ' + str(Accuracy(TP1, TN1, FP1, FN1)) + ' ' +str(JACCARD_INDEX(TP1, FP1, FN1))

print str(Sensitivity(TP, FN)) + ' ' + str(Specifity(TN, FP)) + ' ' + str(Accuracy(TP, TN, FP, FN))+  ' ' +str(JACCARD_INDEX(TP, FP, FN))
