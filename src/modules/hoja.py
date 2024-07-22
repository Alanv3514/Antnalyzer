# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 20:11:16 2024

@author: Vinzon Eric - Quiroga Agustin
"""
import cv2
import math
import numpy as np
from src.modules.KalmanFilter import KalmanFilter

class Aparicion:
    def __init__(self, x, y, xp, yp, hp, area, frame):
        self.x = x
        self.y = y
        self.area = area
        self.frame = frame
        self.xp = xp
        self.yp = yp
        self.hp = hp
        
    def getx(self):
        return self.x
    
    def gety(self):
        return self.y
    
    def getarea(self):
        return self.area
    
    def getframe(self):
        return self.frame
    
    def getxp(self):
        return self.xp
    
    def getyp(self):
        return self.yp
    
    def gethp(self):
        return self.hp
    
    def __str__(self):
        return f"Aparicion: {self.x}, {self.y}, {self.area}"

class Hoja:
    def __init__(self, aparicion, id):
        self.id = id
        self.apariciones = []
        self.addaparicion(aparicion)
     
    def getcantapariciones(self):
        return len(self.apariciones)

    def addaparicion(self , aparicion):
        if len(self.apariciones) < 500:
            self.apariciones.append(aparicion)
        else:
            self.apariciones.pop(0)
            self.apariciones.append(aparicion)
        
    def getarea(self):
        areas = [aparicion.getarea() for aparicion in self.apariciones]
        media= np.mean(areas)
        median = np.percentile(areas, 50)
        q25 = np.percentile(areas, 25)
        q75 = np.percentile(areas, 75)
        
        # Calcular errores
        error_median = np.std(areas)  # Puedes ajustar el cálculo del error según tus necesidades
        error_q25 = np.std(areas) / 2  # Puedes ajustar el cálculo del error según tus necesidades
        error_q75 = np.std(areas) * 1.5  # Puedes ajustar el cálculo del error según tus necesidades

        est = {'media':media,
            'mediana': median,
            'percentil25': q25,
            'percentil75': q75,
            'error_mediana': error_median,
            'error_q25': error_q25,
            'error_q75': error_q75}
        return est
    
    def getID(self):
        return self.id
    
    def __str__(self):
        return f"Hoja {self.id}: areaAcum({self.getarea()}),apariciones({len(self.apariciones)})"
    
    
def xycentro(hoja, tam):
    
    xc=hoja.apariciones[tam-1].getx()
    yc=hoja.apariciones[tam-1].gety()
    return xc, yc

def xypredic(hoja, tam):
    
    xc=hoja.apariciones[tam-1].getxp()
    yc=hoja.apariciones[tam-1].getyp()
    return xc, yc

def posicion(boxes):    #Se le pasa la box detectada y devuelve la posicion correspondiente a esa box
    xy= boxes[0].xyxy[0].astype(int)
    xsup = xy[0]
    ysup = xy[1]
    xinf = xy[2]
    yinf = xy[3]
    ymed=(yinf+ysup)//2
    xmed=(xsup+xinf)//2
    dx=xmed-xsup
    dy=ymed-ysup
    return dx, dy, xmed, ymed


def comparar(dx, dy, xmed, ymed, area, frameactual,gv,kf): # Compara la posicion actual de la hoja con la posicion anterior para reconocer si pertenece o no a una anterior detección
    k=1
    hp=math.hypot(dx, dy)
    centro=(xmed, ymed)
    color = (255, 0, 0)
    #cv2.circle(annotated_frame, centro, int(hp2), color, 2) # Prueba de distancia 
    
    if gv.hojas is None:
        if ymed>gv.yinicio or ymed<gv.yfinal:
            gv.ID+=1
            kf.append(KalmanFilter())
            xp, yp= kf[gv.ID].predict(xmed, ymed)
            apar=Aparicion(xmed, ymed, xp, yp, hp, area, frameactual)
            gv.hojas.append(Hoja(apar, gv.ID))
    else:
        encontrado=0
        for hoja in gv.hojas:
            tam = hoja.getcantapariciones()
            xc, yc = xycentro(hoja, tam)
            if (-k*hp)<xc-xmed<(k*hp) and (-k*hp)<yc-ymed<(k*hp):
                gv.ID=hoja.getID()
                xp, yp= kf[gv.ID].predict(xmed, ymed)
                apar=Aparicion(xmed, ymed, xp, yp, hp, area, frameactual)
                hoja.addaparicion(apar)
                encontrado=1
            elif (-2*k*hp)<xc-xmed<(2*k*hp) and (-2*k*hp)<yc-ymed<(2*k*hp):
                hoja.getID()
                xp, yp= kf[gv.ID].predict(xmed, ymed)
                apar=Aparicion(xmed, ymed, xp, yp, hp, area, frameactual)
                hoja.addaparicion(apar)
                encontrado=1
        
        if encontrado==0:
            if ymed>gv.yinicio or ymed<gv.yfinal:
                gv.ID+=1
                kf.append(KalmanFilter())
                xp, yp= kf[gv.ID].predict(xmed, ymed)
                apar=Aparicion(xmed, ymed, xp, yp, hp, area, frameactual)
                gv.hojas.append(Hoja(apar, gv.ID))    
    
    #cv2.circle(gv.annotated_frame, centro, int(k*hp), color, 2)

