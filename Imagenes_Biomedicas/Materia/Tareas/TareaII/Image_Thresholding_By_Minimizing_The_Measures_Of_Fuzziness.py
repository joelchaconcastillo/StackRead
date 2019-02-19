import math
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

#def Ridler_Calvard(IMG):


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
  [Width, Height] = np.shape(imagen)
  ##two classes..
  Membershipvalue = np.zeros(maxv+1)
  for i in range(minv, maxv):
       kindmean = 0
       if i <= t:
	  kindmean = mu0
       else:
	  kindmean = mu1
       Membershipvalue[i] = 1.0/(1.0 + (abs(i-kindmean)/C))
  return computingEntropy(Membershipvalue, H, minv, maxv)


#Obtaining the image in gray level...
#imagen = misc.imread('BD1/Im001_1.jpg', mode='I')
#imagen = misc.imread('BD1/Im001_1.jpg', flatten=True, mode='I')
imagen = misc.imread('BD1/Im003_1.jpg', flatten=True, mode='I')
imagen2 = misc.imread('BD1/Im003_1.jpg', flatten=True, mode='I')
#imagen2 = misc.imread('BD1/Im001_1.jpg', flatten=True, mode='I')
#imagen = Image.open('BD1/Im001_1.jpg')
#imagen = imagen.convert('L')
imagen = imagen.astype(int)
[Width, Height] = np.shape(imagen)
print imagen
print np.max(imagen)
print np.min(imagen)
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
print opt_threshold
for i in range(0, Width-1):
  for j in range(0, Height-1):
   imagen2[i][j] = minv
   if imagen[i][j] > opt_threshold:
    imagen2[i][j] = maxv

fig, (uno, dos) = plt.subplots(1,2)
uno.imshow(imagen, cmap='gray')
dos.imshow(imagen2, cmap='gray');
print imagen
plt.show()

#imagen2.save('img_gris.png')
