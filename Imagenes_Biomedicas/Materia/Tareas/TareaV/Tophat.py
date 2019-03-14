import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

#from skimage.morphology import square

fprM = np.zeros(255)
tprM = np.zeros(255)
P = 0
N = 0

for nim in range (1, 21):
 imgref = cv2.imread('BD_20_Angios/'+str(nim)+'_gt.png',0)
 img = cv2.imread('BD_20_Angios/'+str(nim)+'.png',0)
 #kernel =  np.ones((5,5),np.uint8)# cv2.getStructuringElement(cv2.MORPH_,(19,19))
 kernel = cv2.getStructuringElement(0,(19,19))
 #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19,19))
  #sz = 19
 #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*sz-1, 2*sz-1))
 #img = cv2.erode(mask,kernel,iterations = 1)
 img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
#ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
 imgref[imgref>0]=1
 P += np.sum(imgref.flatten())
 N += imgref.flatten().size - P
 for i in range(0,255):
  ret2,th2 = cv2.threshold(img,i,255, cv2.THRESH_BINARY)
  th2[th2>0]=1
  tn, fp, fn, tp = metrics.confusion_matrix(imgref.flatten(), th2.flatten()).ravel()
  fprM[i] += fp
  tprM[i] += tp

print metrics.auc(fprM/(N), tprM/(P), reorder=True)

#cv2.imshow('th2',th2)
#cv2.waitKey(0)
#cv2.destroyAllWindows() 



####blur = cv2.GaussianBlur(img,(5,5),0)
####ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
####cv2.imshow('img',img)
####th2[th2 >0]=1
####imgref[imgref >0]=1
###th2 /=255
###imgref /=255

###roc_auc = dict()
###
####fpr1, tpr1, thresholds = metrics.roc_curve(th2.flatten(), imgref.flatten())
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
