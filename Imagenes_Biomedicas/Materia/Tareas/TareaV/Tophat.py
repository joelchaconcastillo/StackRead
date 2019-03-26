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
  for nim in range (1, 2):
   imgref = cv2.imread('BD_20_Angios/'+str(nim)+'_gt.png',0).flatten()
   img = cv2.imread('BD_20_Angios/'+str(nim)+'.png',0)

   img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
  

  # ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
   imgref=imgref/255
   indexes = np.unique(img.flatten())
   scores = np.zeros(imgref.size)
  # P += np.sum(imgref.flatten())
  # N += imgref.flatten().size - P
   for i in range(0,255):
   #for k in range(0,indexes.size):
#    i = indexes[k]
#    print i
    ret2,th2 = cv2.threshold(img,i,255, cv2.THRESH_BINARY)
    th2 =th2.flatten()/255
    #th2 = np.copy(img)

   # th2 = np.copy(img.flatten())
   # th2[th2<=i]=0
   # th2[th2>i]=1
  #  plt.imshow(th2.reshape(300,300), 'gray')
  #  plt.show()

    tp = np.sum(np.logical_and(th2, imgref))
    tn = np.sum(np.logical_and(np.logical_not(th2), np.logical_not(imgref)))
    fn = np.sum(np.logical_and(np.logical_not(th2), imgref))
    fp = np.sum(np.logical_and(th2, np.logical_not(imgref)))
    #tn, fp, fn, tp = metrics.confusion_matrix(imgref.flatten(), th2.flatten()).ravel()
#    print str(tp)+ " " + str(tn) + " " +str(fp)+" "+str(fn)
    TPFN[i] += tp+fn
    FPTN[i] += fp+tn
    FP[i] += fp
    TP[i] += tp
#    print TP[i]/TPFN[i]
#  plt.plot(fprM/(N), tprM/(P))
#  plt.ylim(0, 1.1)
#  plt.xlim(0, 1.1)
#  plt.show()
#  print -metrics.auc(fprM/(N+0.0000001), tprM/(P+0.000001), reorder=True)
  #return auc_from_fpr_tpr(fprM/(N+0.0000001), tprM/(P+0.000001), True)
#  print auc_from_fpr_tpr(FP/(FPTN+0.01), TP/(TPFN+0.01), True)
#  exit(0)
  return auc_from_fpr_tpr(FP/(FPTN+0.01), TP/(TPFN+0.01), True)
#  print metrics.roc_auc_score(imgref.flatten(),  scores/indexes.size)
#  return -metrics.auc(fprM/(N+0.0000001), tprM/(P+0.000001), reorder=True)

def improving(kernel, obj):
   bestkernel = np.copy(kernel)
   bestobj = obj
   size = kernel.size
   side = int(math.sqrt(size))
   selected = np.zeros(size)
   print evaluateKernel(kernel.reshape(int(math.sqrt(size)),int(math.sqrt(size))))
   window = np.random.randint(0, 10)
   for k in range(0,100):
     current = np.copy(bestkernel)
     currentobj = bestobj
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
     currentobj = evaluateKernel(current.reshape(side,side))
     if currentobj < bestobj:
        k=0
	bestkernel = np.copy(bestkernel)
	bestobj = currentobj 
        print "improved  " + str(currentobj)
     else:
	window = np.random.randint(0, 10)
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
   print img[0]
   print wiseproduct[0]
   pA = wiseproduct / wiseproduct.sum()
   return np.sum(pA*np.log2(pA+0.000001))

#Initialize probability vector...
size = 19
#kernel = cv2.getStructuringElement(2,(size,size)).flatten()
pop = 20
NBest = 5
sizeStructure = size*size
kernel = np.zeros((pop, sizeStructure))
bestkernel = np.zeros(sizeStructure)
prob = np.ones(sizeStructure)*0.5
rocvalues = np.zeros(pop)
maxite = 300
for k in range(0,3):
 kernel[k,:] = cv2.getStructuringElement(k,(size,size)).flatten()
 rocvalues[k] = evaluateKernel(kernel[k].astype(int).reshape(size, size))
 #rocvalues[k] = evaluateKernelEntropy(kernel[k].astype(int).reshape(size, size))
 print rocvalues[k]

for i in range(0, maxite):
  for k in range(4,pop):
    #Sampling...
    randomN = np.random.uniform(0,1, sizeStructure)
    kernel[k, randomN < prob] = 1
    kernel[k, randomN >= prob] = 0
    #rocvalues[k] = evaluateKernel(kernel.reshape(size, size))
    rocvalues[k] = evaluateKernel(kernel[k].astype(int).reshape(size, size))
    print str(rocvalues[k])+ " " +str(k)
  #improving 
  #for k in range(0,pop):
  #  print "before.. " + str(rocvalues[k])+ " " +str(k)
  #  kernel[k,: ], rocvalues[k] = improving(kernel[k,:], rocvalues[k])
  #  print "after.."+str(rocvalues[k])+ " " +str(k)
#  print kernel[3,:]
  #Select the bests roc curve...
  bestIndexes = np.argsort(rocvalues)
  #learning probabilities...
  prob = np.sum(kernel[bestIndexes[0:NBest],:], axis=0  ) / NBest
#  prob = [  np.random.normal(prob[l], 0.001*(1-(i/maxite)))   for l in range(0, prob.size)]
#  prob[prob>1]=1
#  prob[prob<0]=0
  #saving elite
  kernel[3,:] = np.copy(kernel[bestIndexes[0]])
  rocvalues[3] = rocvalues[bestIndexes[0]]

  print "best... " + str( rocvalues[3])
  print kernel[3,:]

#for i in range(0,30):
#  for k in range(0,3):
#   kernel[k,:] = cv2.getStructuringElement(k,(size,size)).flatten()
#   rocvalues[k] = evaluateKernel(kernel[k].astype(int).reshape(size, size))
#   #rocvalues[k] = evaluateKernelEntropy(kernel[k].astype(int).reshape(size, size))
#   print rocvalues[k]
#  for k in range(4,pop):
#    #Sampling...
#    randomN = np.random.uniform(0,1, sizeStructure)
#    kernel[k, randomN < prob] = 1
#    kernel[k, randomN >= prob] = 0
#    #rocvalues[k] = evaluateKernel(kernel.reshape(size, size))
#    rocvalues[k] = evaluateKernel(kernel[k].astype(int).reshape(size, size))
#    #rocvalues[k] = evaluateKernelEntropy(kernel[k].astype(int).reshape(size, size))
#    print str(rocvalues[k])+" "+str(k)
#    for k in range(0,pop):
#     kernel[k], rocvalues[k] = improving(kernel[k], rocvalues[k])
#  #Select the bests roc curve...
#  bestIndexes = np.argsort(rocvalues)
#  #learning probabilities...
#  prob = np.sum(kernel[bestIndexes[0:NBest],:], axis=0  ) / NBest
#  kernel[3,:] = np.copy(kernel[bestIndexes[0]])
#  print kernel[3,:]
#  rocvalues[3] = rocvalues[bestIndexes[0]]
#  print "best... " + str(rocvalues[bestIndexes[0]])
#  print kernel[3,:]
#print "best... " + str( evaluateKernel(kernel[3].reshape(size, size))  )
#print kernel[3,:]




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
