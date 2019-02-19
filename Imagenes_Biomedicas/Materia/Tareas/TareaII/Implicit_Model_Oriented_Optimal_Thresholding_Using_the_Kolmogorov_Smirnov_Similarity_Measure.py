from scipy import misc
import matplotlib.pyplot as plt

#def Ridler_Calvard(IMG):



imagen = misc.imread('BD1/Im001_1.jpg')
fig, (uno, dos) = plt.subplots(1,2)
uno.imshow(imagen)
imagen = 255-imagen
dos.imshow(imagen);
plt.show()
