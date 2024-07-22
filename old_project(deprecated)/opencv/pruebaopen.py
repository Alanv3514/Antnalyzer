import cv2
import numpy as np

font=cv2.FONT_HERSHEY_SIMPLEX
limitemenor= [25, 52, 72]
limitemayor= [102, 255, 255]

menor=np.array(limitemenor, dtype="uint8")
mayor=np.array(limitemayor, dtype="uint8")


img = cv2.imread(r"C:\Users\Agustin\Documents\Python\opencv\assets\logo.jpg", 0)
img2= cv2.imread(r"C:\Users\Agustin\Documents\Python\opencv\assets\hormigas_test.jpg", -1)

img2hsv=cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
mask=cv2.inRange(img2hsv, menor, mayor)

kernal = np.ones((2,2), "uint8")
#mask=cv2.dilate(mask, kernal)
salida=cv2.bitwise_and(img2, img2, mask=mask)
contorno, jerarquia = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img2, contorno, -1, (0,255,0), 3)

for c in contorno:
    area=cv2.contourArea(c)
    if area > 80:
        x,y,w,h=cv2.boundingRect(c)
        #nuevContorno=cv2.convexHull(c)
        cv2.rectangle(img2,(x,y), (x+w,y+h), (0, 255, 0), 1)
        cv2.putText(img2, 'Hoja',(x-3,y-3),font, 0.75, (0, 255, 0), 1, cv2.LINE_AA)
        #cv2.drawContours(img2, [c], 0, (0,255,0), 2)

#cv2.imwrite('salida_imagen.jpg', img)

cv2.imshow("Imagen", img2)
cv2.imshow("Imagen1", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

