import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

#from skimage.morphology import square


def evaluateKernel(kernel):
  fprM = np.zeros(255)
  tprM = np.zeros(255)
  P = np.zeros(255)
  N = np.zeros(255)
  kernel = np.uint8(kernel)
#  kernel =  np.ones((49,49),np.uint8)# cv2.getStructuringElement(cv2.MORPH_,(19,19))
#  kernel = cv2.getStructuringElement(2,(29,29))
  for nim in range (1, 2):
   imgref = cv2.imread('BD_20_Angios/'+str(nim)+'_gt.png',0).flatten()
   img = cv2.imread('BD_20_Angios/'+str(nim)+'.png',0)
   img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
  
   plt.imshow(img, 'gray')
   plt.show()
  # ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
   imgref=imgref/255
   indexes = np.unique(img.flatten())
  # P += np.sum(imgref.flatten())
  # N += imgref.flatten().size - P
   #for i in range(0,255):
   for k in range(0,indexes.size):
    i = indexes[k]
   # ret2,th2 = cv2.threshold(img,i,255, cv2.THRESH_BINARY)
    #th2 = np.copy(img)
    th2 = img.flatten()
    th2[th2<i]=0
    th2[th2>=i]=1
#    tn = 0
#    tp = 0
#    fn = 0
#    fp = 0
#    for z in range(0, imgref.size):
#       if imgref[z] == 1 & th2[z] == 1:
#	tp += 1
#       if imgref[z] == 0 & th2[z] == 0:
#        tn +=1
#       if imgref[z] == 0 & th2[z] == 1:
#        fp +=1   
#       if imgref[z] == 1 & th2[z] == 0:
#        fn +=1
    tn, fp, fn, tp = metrics.confusion_matrix(imgref.flatten(), th2.flatten()).ravel()
    P[i] += tp+fn
    N[i] += fp+tn
    fprM[i] += fp
    tprM[i] += tp
#  plt.plot(fprM/(N), tprM/(P))
#  plt.ylim(0, 1.1)
#  plt.xlim(0, 1.1)
#  plt.show()
  return -metrics.auc(fprM/(N+0.0000001), tprM/(P+0.000001), reorder=True)


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
size = 39
#kernel = cv2.getStructuringElement(2,(size,size)).flatten()
pop = 10
NBest = 5
sizeStructure = size*size
kernel = np.zeros((pop, sizeStructure))
bestkernel = np.zeros(sizeStructure)
prob = np.ones(sizeStructure)*0.5
rocvalues = np.zeros(pop)

for i in range(0,30):
  for k in range(0,3):
   kernel[k,:] = cv2.getStructuringElement(k,(size,size)).flatten()
   rocvalues[k] = evaluateKernel(kernel[k].astype(int).reshape(size, size))
   #rocvalues[k] = evaluateKernelEntropy(kernel[k].astype(int).reshape(size, size))
   print rocvalues[k]
  for k in range(3,pop):
    #Sampling...
    randomN = np.random.uniform(0,1, sizeStructure)
    kernel[k, randomN < prob] = 1
    kernel[k, randomN >= prob] = 0
    #rocvalues[k] = evaluateKernel(kernel.reshape(size, size))
    rocvalues[k] = evaluateKernel(kernel[k].astype(int).reshape(size, size))
    #rocvalues[k] = evaluateKernelEntropy(kernel[k].astype(int).reshape(size, size))
    print rocvalues[k]
  #Select the bests roc curve...
  bestIndexes = np.argsort(rocvalues)
  #learning probabilities...
  prob = np.sum(kernel[bestIndexes[0:NBest],:], axis=0  ) / NBest
  kernel[3,:] = np.copy(kernel[bestIndexes[0]])
  print kernel[3,:]
  rocvalues[3] = rocvalues[bestIndexes[0]]
  print "best... " + str(rocvalues[bestIndexes[0]])
  print kernel[3,:]
print "best... " + str( evaluateKernel(kernel[0].reshape(size, size))  )
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
