import cv2
import numpy as np
from rastreador import *

seguimiento = Rastreador()

#Prueba de programa de video
font=cv2.FONT_HERSHEY_SIMPLEX
limitemenor= [35, 25, 35]
limitemayor= [85, 255, 255]

menor=np.array(limitemenor, dtype="uint8")
mayor=np.array(limitemayor, dtype="uint8")
cap=cv2.VideoCapture(r"C:\Users\Agustin\Videos\test_1.mp4")
frametime= 16
detector= cv2.createBackgroundSubtractorMOG2(history=100000)

while True:
    ret, img2= cap.read()
    roi = img2[100: 350, 200:420] #Region de interes

    img2hsv=cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    mask_1=detector.apply(roi)
    #_, mask_1= cv2.threshold(mask_1, 128, 255, cv2.THRESH_BINARY)
    mask_1 = cv2.erode(mask_1, np.ones((3, 3), dtype=np.uint8))
    mask=cv2.inRange(img2hsv, menor, mayor)
    contorno, jerarquia = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detecciones = []
    for c in contorno:
        area=cv2.contourArea(c)
        if area > 400:
            x,y,w,h=cv2.boundingRect(c)
            cv2.rectangle(roi,(x,y), (x+w,y+h), (0, 255, 0), 1)
            detecciones.append([x,y,w,h])
            #cv2.putText(img2, 'Hoja',(x-3,y-3),font, 0.75, (0, 255, 0), 1, cv2.LINE_AA)
            #cv2.drawContours(roi, [nuevContorno], 0, (0,255,0), 2)
    print(detecciones)
    #Seguimiento de las detecciones
    info_id = seguimiento.rastreo(detecciones)
    for inf in info_id:
        x, y, w, h, id= inf
        cv2.putText(roi, str(id), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255),2)
    cv2.imshow('Mask', mask_1)
    cv2.imshow('Region de interes',roi)
    if cv2.waitKey(frametime) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()