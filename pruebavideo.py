import cv2
import numpy as np

font=cv2.FONT_HERSHEY_SIMPLEX
limitemenor= [35, 25, 35]
limitemayor= [85, 255, 255]

menor=np.array(limitemenor, dtype="uint8")
mayor=np.array(limitemayor, dtype="uint8")
cap=cv2.VideoCapture(0)

while True:
    ret, img2= cap.read()
    img2hsv=cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(img2hsv, menor, mayor)
    contorno, jerarquia = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contorno:
        area=cv2.contourArea(c)
        if area > 150:
            x,y,w,h=cv2.boundingRect(c)
            nuevContorno=cv2.convexHull(c)
            #cv2.rectangle(img2,(x,y), (x+w,y+h), (0, 255, 0), 1)
            #cv2.putText(img2, 'Hoja',(x-3,y-3),font, 0.75, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.drawContours(img2, [nuevContorno], 0, (0,255,0), 2)
    cv2.imshow('Frame', img2)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()