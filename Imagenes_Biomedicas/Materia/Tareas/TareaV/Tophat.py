from __future__ import print_function
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
import itertools
import operator
import math

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
#from skimage.morphology import square
def auc_from_fpr_tpr(fpr, tpr, trapezoid=False):
    inds = [i for (i, (s, e)) in enumerate(zip(fpr[: -1], fpr[1: ])) if s != e] + [len(fpr) - 1]
    fpr, tpr = fpr[inds], tpr[inds]
    area = 0
    ft = zip(fpr, tpr)
    for p0, p1 in zip(ft[: -1], ft[1: ]):
        area += (p1[0] - p0[0]) * ((p1[1] + p0[1]) / 2 if trapezoid else p0[1])
    return area

def evaluateKernel(kernel):
  FP = np.zeros(255)
  TP = np.zeros(255)
  FPTN = np.zeros(255)
  TPFN = np.zeros(255)
  kernel = np.uint8(kernel)
#  kernel = cv2.getStructuringElement(2,(29,29))
  for nim in range (1, 2):
   imgref = cv2.imread('BD_20_Angios/'+str(nim)+'_gt.png',0).flatten()
   img = cv2.imread('BD_20_Angios/'+str(nim)+'.png',0)
   for l in range(0,3):
    img = cv2.imread('BD_20_Angios/'+str(nim)+'.png',0)
#    kernel =  cv2.getStructuringElement(0,(19,19))
    img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
 #   ret2,img= cv2.threshold(img,10,255, cv2.THRESH_BINARY)
  #  plt.imshow(img, cmap='gray')
   # plt.show()

   
  

   #ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
  # for i in range(1, 255):
  #  ret2,th2 = cv2.threshold(img,i,255, cv2.THRESH_BINARY)
  #  plt.imshow(th2, cmap='gray')
  #  plt.show()

   imgref=imgref/255
   indexes = np.unique(img.flatten())
   
  minv = 100000000
  tt = -1
  for i in range(0, 255):
    ret2,th2 = cv2.threshold(img,i,255, cv2.THRESH_BINARY)
    th2 =th2.flatten()/255
    tp = np.sum(np.logical_and(th2, imgref))
    tn = np.sum(np.logical_and(np.logical_not(th2), np.logical_not(imgref)))
    fn = np.sum(np.logical_and(np.logical_not(th2), imgref))
    fp = np.sum(np.logical_and(th2, np.logical_not(imgref)))
    TPFN[i] += tp+fn+0.0
    FPTN[i] += fp+tn+0.0
    FP[i] += fp+0.0
    TP[i] += tp+0.0
    sensitivity = (tp/float(tp+fn))
    specifity = (fp/float(fp+tn))
    dis = math.sqrt(  (1.0-sensitivity)*(1.0-sensitivity)  + (specifity)*(specifity))
    if dis < minv:
       minv = dis
       tt = i 
  FPR = np.append(FP/FPTN,[0.0,1.0])
  TPR = np.append(TP/TPFN,[1.0,0.0])
#  print(TP/(TPFN))
#  print(FP/(FPTN))
#  print("------")
#  print(np.trapz( FPR,TPR))
 # return auc_from_fpr_tpr(FPR, TPR, True)
  return -metrics.auc(FPR, TPR, reorder=True), tt

def evaluateKernelbyImage(kernel, Nsample):
  FP = np.zeros(255)
  TP = np.zeros(255)
  FPTN = np.zeros(255)
  TPFN = np.zeros(255)
  kernel = np.uint8(kernel)
#  kernel =  np.ones((49,49),np.uint8)# cv2.getStructuringElement(cv2.MORPH_,(19,19))
#  kernel = cv2.getStructuringElement(2,(29,29))
  for Lnim in range (0, Nsample.size):
   nim = Nsample[Lnim]
   imgref = cv2.imread('BD_20_Angios/'+str(nim)+'_gt.png',0).flatten()
   img = cv2.imread('BD_20_Angios/'+str(nim)+'.png',0)

   img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
  

  # ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
   imgref=imgref/255
   indexes = np.unique(img.flatten())
   
   for i in range(0, 255):
    ret2,th2 = cv2.threshold(img,i,255, cv2.THRESH_BINARY)
    th2 =th2.flatten()/255
    tp = np.sum(np.logical_and(th2, imgref))
    tn = np.sum(np.logical_and(np.logical_not(th2), np.logical_not(imgref)))
    fn = np.sum(np.logical_and(np.logical_not(th2), imgref))
    fp = np.sum(np.logical_and(th2, np.logical_not(imgref)))
    TPFN[i] += tp+fn
    FPTN[i] += fp+tn
    FP[i] += fp
    TP[i] += tp
  #return auc_from_fpr_tpr(fprM/(N+0.0000001), tprM/(P+0.000001), True)
  return -metrics.auc(FP/(FPTN+0.01), TP/(TPFN+0.01), reorder=True)

def improving(kernel, obj):
   bestkernel = np.copy(kernel)
   bestobj = obj
   size = kernel.size
   side = int(math.sqrt(size))
   selected = np.zeros(size)
   window = np.random.randint(0, 5)
   indexImage = np.random.randint(1, 21, 2)
   bestobj= evaluateKernelbyImage(kernel.reshape(side,side), indexImage)
   for k in range(0,300):
     current = np.copy(bestkernel)
     currentobj = bestobj
     for z in range(0, np.random.randint(0, 2)):
      bit = np.random.randint(0, 2)
   #select a random bit..
      x = np.random.randint(0, side)
      y = np.random.randint(0, side)
      while selected[(x*side+y)] ==1:
       x = np.random.randint(0, side)
       y = np.random.randint(0, side)
      selected[(x*side+y)] ==1
      
      for r in range(x, x+window):
        for c in range(y, y+window):
          choords = ((r%side)*side + (c%side))
          if current[choords ] == 1:
            current[choords ] = 0
          else:
    	    current[choords ] = 1
    	    current[choords ] = bit
     currentobj = evaluateKernelbyImage(current.reshape(side,side), indexImage)
     if currentobj < bestobj:
        k=0
	bestkernel = np.copy(current)
	bestobj = currentobj 
        eprint ("improved  " + str(currentobj))
     else:
	window = np.random.randint(0, 5)
   t=0
   bestobj,t = evaluateKernel(bestkernel.reshape(side,side))
   eprint ("best..."+str(bestobj))
   if bestobj < obj:
    return bestkernel, bestobj
   else :
    return kernel, obj

def evaluateKernelEntropy(kernel):
  fprM = np.zeros(255)
  tprM = np.zeros(255)
  P = np.zeros(255)
  N = np.zeros(255)
  kernel = np.uint8(kernel)
#  kernel =  np.ones((49,49),np.uint8)# cv2.getStructuringElement(cv2.MORPH_,(19,19))
#  kernel = cv2.getStructuringElement(2,(29,29))
  for nim in range (1, 2):
   img = cv2.imread('BD_20_Angios/'+str(nim)+'.png',0)
   img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel).flatten()
   ##sobel...
   dx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
   dy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
   mag = np.hypot(dx, dy)  # magnitude
   mag *= 255.0 / np.max(mag)  # normalize (Q&D)
   wiseproduct = np.absolute(np.multiply(img, mag.flatten()))
   pA = wiseproduct / wiseproduct.sum()
   return np.sum(pA*np.log2(pA+0.000001))

#Initialize probability vector...
size = 50
#kernel = cv2.getStructuringElement(2,(size,size)).flatten()
pop = 10
NBest = 5
sizeStructure = size*size
kernel = np.zeros((pop, sizeStructure))
bestkernel = np.zeros(sizeStructure)
prob = np.ones(sizeStructure)*0.5
rocvalues = np.zeros(pop)
tt = np.zeros(pop)
maxite = 30
for k in range(0,1): 
 kernel[k,:] = cv2.getStructuringElement(k,(size,size)).flatten()
 print (kernel[k].reshape(size,size))
 rocvalues[k], tt[k]= evaluateKernel(kernel[k,:].astype(int).reshape(size, size))
 eprint (rocvalues[k])
for i in range(0, maxite):
  eprint (i)
  for k in range(1,pop):
    #Sampling...
    randomN = np.random.uniform(0,1, sizeStructure)
    kernel[k, randomN < prob] = 1
    kernel[k, randomN >= prob] = 0
    rocvalues[k],tt[k] = evaluateKernel(kernel[k].astype(int).reshape(size, size))
#    rocvalues[k] = evaluateKernelEntropy(kernel[k].astype(int).reshape(size, size))
  #improving 
  for k in range(0,pop):
    eprint (rocvalues[k])
    kernel[k,: ], rocvalues[k] = improving(kernel[k,:], rocvalues[k])
    eprint (rocvalues[k])
  #Select the bests roc curve...
  bestIndexes = np.argsort(rocvalues)
  #learning probabilities...
  prob = np.sum(kernel[bestIndexes[0:NBest],:], axis=0  ) / NBest
  #saving elite
  kernel[3,:] = np.copy(kernel[bestIndexes[0],:])
  rocvalues[3] = rocvalues[bestIndexes[0]]
  tt[3] = tt[bestIndexes[0]]
  eprint (str( rocvalues[bestIndexes]))
  eprint ("------ " + str( rocvalues[3]))
eprint (evaluateKernel(kernel[3].astype(int).reshape(size, size)))
eprint (str( rocvalues[3]))
eprint (kernel[3,:])




img = cv2.imread('BD_20_Angios/1.png',0)
img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, np.uint8(kernel[3,:]).reshape(size,size))
ret2,img= cv2.threshold(img,tt[3],255, cv2.THRESH_BINARY)
plt.imshow(img, cmap='gray')
plt.show()



#print evaluateKernel(kernel)
#print kernel


#evaluateNeighbour

## cv2.imshow('th2',th2)
## cv2.waitKey(0)
## cv2.destroyAllWindows() 



####blur = cv2.GaussianBlur(img,(5,5),0)
####ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
####cv2.imshow('img',img)
####th2[th2 >0]=1
####imgref[imgref >0]=1
###th2 /=255
###imgref /=255

###roc_auc = dict()
###
#fpr1, tpr1, thresholds = metrics.roc_curve(th2.flatten(), imgref.flatten())
###
#plt.plot(fpr, tpr)
#plt.show()
###print metrics.auc(fpr1, tpr1)
###
####fpr["micro"], tpr["micro"], _ = metrics.roc_curve(th2.flatten(), imgref.flatten())
###print metrics.roc_auc_score(th2.flatten(), imgref.flatten())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#cv2.imshow('th2',th2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#plt.imshow(th2, 'gray')
#plt.show()
