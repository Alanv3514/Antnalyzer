import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

font=cv2.FONT_HERSHEY_SIMPLEX
limitemenor= [25, 52, 72]
limitemayor= [102, 255, 255]

menor=np.array(limitemenor, dtype="uint8")
mayor=np.array(limitemayor, dtype="uint8")


img= cv2.imread(r"C:\Users\Agustin\Documents\Python\opencv\assets\hormigas_test.jpg", -1)

b,g,r,a=cv2.split(img)

plt.figure(figsize=(10,8))

plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('RGB')

plt.subplot(2,2,2)
plt.imshow(r, cmap='gray')
plt.title('Rojo')

plt.subplot(2,2,3)
plt.imshow(g, cmap='gray')
plt.title('Verde')

plt.subplot(2,2,4)
plt.imshow(b, cmap='gray')
plt.title('Azul')
plt.show()

print(g.max())
print(r.max())

#NDI=(G - R)/(G+R)
max=(g.max()/2).astype('uint8')
print(max)
new_g=(0.5 * g).astype('uint8')
new_g=new_g + (max * np.ones(new_g.shape)).astype('uint8') #va de 127 a 255

new_r =(0.5*r).astype('uint8') #va de 0 a 127

g[(g==0)]=2
r[(r==0)]=2
g[(g==1)]=2
r[(r==1)]=2
suma = (0.5 * new_g).astype('uint8') + new_r

#suma[(suma==0)]=1 #Evitamos que haya una división por 0

resta = new_g-new_r
print(resta.max())
plt.figure()

plt.subplot(1,2,1)
plt.imshow(resta, cmap='gray')
plt.title('Resta')

plt.subplot(1,2,2)
plt.imshow(suma, cmap='gray')
plt.title('Suma')
plt.show()

NDI= np.divide(resta.astype('float32'),suma.astype('float32'))
print(NDI)
print(NDI.dtype)
print(NDI.max())

plt.figure()
plt.imshow(NDI, cmap='gray')
plt.show()


NDI_scaled=NDI.copy()
NDI_scaled= NDI_scaled - NDI.min()
NDI_scaled = NDI_scaled * (255/NDI.max())
NDI_int = NDI_scaled.astype('uint8')

#Hacemos thresholding
umbral=1.0
ret2, th1 = cv2.threshold(NDI_int, umbral, 255.0, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#Hacemos la división, el rango de division será entre [0,1]

shape = th1.shape
height = int(shape[0] / 2)
width = int(shape[1] / 2)
image = cv2.resize(th1, (width, height))
cv2.imshow('NDI', image)
cv2.waitKey(0)
cv2.destroyAllWindows()