import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('BD_20_Angios/1.png',0)
#kernel =  np.ones((5,5),np.uint8)# cv2.getStructuringElement(cv2.MORPH_,(19,19))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(19,19))
img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)


ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('th2',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.imshow(th2, 'gray')
plt.show()
