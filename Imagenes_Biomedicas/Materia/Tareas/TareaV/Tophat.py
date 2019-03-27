import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
import itertools
import operator
import math
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
#  kernel =  np.ones((49,49),np.uint8)# cv2.getStructuringElement(cv2.MORPH_,(19,19))
#  kernel = cv2.getStructuringElement(2,(29,29))
  for nim in range (1, 21):
   imgref = cv2.imread('BD_20_Angios/'+str(nim)+'_gt.png',0).flatten()
   img = cv2.imread('BD_20_Angios/'+str(nim)+'.png',0)

   img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
  

  # ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
   imgref=imgref/255
   indexes = np.unique(img.flatten())
   
   for i in range(0, 254):
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

def evaluateKernelbyImage(kernel, index):
  FP = np.zeros(255)
  TP = np.zeros(255)
  FPTN = np.zeros(255)
  TPFN = np.zeros(255)
  kernel = np.uint8(kernel)
#  kernel =  np.ones((49,49),np.uint8)# cv2.getStructuringElement(cv2.MORPH_,(19,19))
#  kernel = cv2.getStructuringElement(2,(29,29))
  for nim in range (index, index+1):
   imgref = cv2.imread('BD_20_Angios/'+str(nim)+'_gt.png',0).flatten()
   img = cv2.imread('BD_20_Angios/'+str(nim)+'.png',0)

   img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
  

   imgref=imgref/255
   scores = np.zeros(imgref.size)
  # P += np.sum(imgref.flatten())
  # N += imgref.flatten().size - P
   for i in range(0,255):
    ret2,th2 = cv2.threshold(img,i,255, cv2.THRESH_BINARY)
    th2 =th2.flatten()/255
    tp = np.sum(np.logical_and(th2, imgref))
    tn = np.sum(np.logical_and(np.logical_not(th2), np.logical_not(imgref)))
    fn = np.sum(np.logical_and(np.logical_not(th2), imgref))
    fp = np.sum(np.logical_and(th2, np.logical_not(imgref)))
    #tn, fp, fn, tp = metrics.confusion_matrix(imgref.flatten(), th2.flatten()).ravel()
    TPFN[i] += tp+fn
    FPTN[i] += fp+tn
    FP[i] += fp
    TP[i] += tp
#  print -metrics.auc(fprM/(N+0.0000001), tprM/(P+0.000001), reorder=True)
  #return auc_from_fpr_tpr(fprM/(N+0.0000001), tprM/(P+0.000001), True)
#  print auc_from_fpr_tpr(FP/(FPTN+0.01), TP/(TPFN+0.01), True)
  return auc_from_fpr_tpr(FP/(FPTN+0.01), TP/(TPFN+0.01), True)
#  print metrics.roc_auc_score(imgref.flatten(),  scores/indexes.size)
#  return -metrics.auc(fprM/(N+0.0000001), tprM/(P+0.000001), reorder=True)


def improving(kernel, obj):
   bestkernel = np.copy(kernel)
   bestobj = obj
   size = kernel.size
   side = int(math.sqrt(size))
   selected = np.zeros(size)
   window = np.random.randint(0, 5)
   for k in range(0,300):
     current = np.copy(bestkernel)
     currentobj = bestobj
     for z in range(0, np.random.randint(0, 3)):
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
     currentobj = evaluateKernel(current.reshape(side,side))
     if currentobj < bestobj:
        k=0
	bestkernel = np.copy(current)
	bestobj = currentobj 
        print "improved  " + str(currentobj)
     else:
	window = np.random.randint(0, 5)
   return bestkernel, bestobj

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
size = 19
#kernel = cv2.getStructuringElement(2,(size,size)).flatten()
pop = 10
NBest = 5
sizeStructure = size*size
kernel = np.zeros((pop, sizeStructure))
bestkernel = np.zeros(sizeStructure)
prob = np.ones(sizeStructure)*0.5
rocvalues = np.zeros(pop)
maxite = 100
for k in range(0,3): 
 print k
 kernel[k,:] = cv2.getStructuringElement(k,(size,size)).flatten()
 rocvalues[k] = evaluateKernel(kernel[k].astype(int).reshape(size, size))
 print rocvalues[k]
 exit(0)
for i in range(0, maxite):
  print i
  for k in range(4,pop):
    #Sampling...
    randomN = np.random.uniform(0,1, sizeStructure)
    kernel[k, randomN < prob] = 1
    kernel[k, randomN >= prob] = 0
    rocvalues[k] = evaluateKernel(kernel[k].astype(int).reshape(size, size))
#    rocvalues[k] = evaluateKernelEntropy(kernel[k].astype(int).reshape(size, size))
  #improving 
  #for k in range(0,pop):
  #  kernel[k,: ], rocvalues[k] = improving(kernel[k,:], rocvalues[k])
  #Select the bests roc curve...
  bestIndexes = np.argsort(rocvalues)
  #learning probabilities...
  prob = np.sum(kernel[bestIndexes[0:NBest],:], axis=0  ) / NBest
  #saving elite
  kernel[3,:] = np.copy(kernel[bestIndexes[0],:])
  rocvalues[3] = rocvalues[bestIndexes[0]]
  print str( rocvalues[bestkernel])
  print "------ " + str( rocvalues[3])
print evaluateKernel(kernel[3].astype(int).reshape(size, size))
print str( rocvalues[3])
print kernel[3,:]


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
