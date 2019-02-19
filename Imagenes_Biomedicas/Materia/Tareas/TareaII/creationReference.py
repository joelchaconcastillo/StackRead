import math
import sys
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


infile = str(sys.argv[1])+'.jpg'

imagen = misc.imread(infile, flatten=True, mode='I')
imagenReference = misc.imread(infile, flatten=True, mode='I')
imagen = imagen.astype(int)

imagenReference[:][:]=0
checked = imagenReference
sys.setrecursionlimit(1500000000)
def exploreImage(x, y, grayref, cont):
  [Width, Height] = np.shape(imagen)
  #cases base...
  if x < 0:
   return 0
  if x >= Width-1:
   return 0
  if y < 0: 
   return
  if y >= Height-1:
   return 0
  if abs(grayref-imagen[x][y]) > 25:
   return 0
  if imagenReference[x][y] == 254:
   return 0
  if cont > 10000:
   return 0
  if checked[x][y] ==1:
   return 0
  imagenReference[x][y] = 254
  checked[x][y]=1
  exploreImage(x+1, y,  grayref, cont+1)
  exploreImage(x, y+1,  grayref, cont+1)
  exploreImage(x-1, y,  grayref, cont+1)
  exploreImage(x, y-1,  grayref, cont+1)
  return 0

[Width, Height] = np.shape(imagen)
f = open(str(sys.argv[1])+'.xyc', "r")
#in each coordinate...
for x in f:
  row = x.split()
  exploreImage(int(row[1]), int(row[0]), imagen[int(row[1])][int(row[0])], 0) 

fig, (uno, dos) = plt.subplots(1,2)
uno.imshow(imagen, cmap='gray')
dos.imshow(imagenReference, cmap='gray');
plt.show()
#misc.imsave(str(sys.argv[2]), imagenReference)
