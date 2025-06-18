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
        self.match_count = 0
    
    def incrementar_bandera(self):
        self.match_count += 1 

    def get_bandera(self):
        return self.match_count
        
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
        self.valid_id = None
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
        maximo = np.max(areas)
        minimo = np.min(areas)
        # Calcular errores
        error_median = np.std(areas)   
        error_q25 = np.std(areas) / 2   
        error_q75 = np.std(areas) * 1.5  

        est = {'media':media,
            'mediana': median,
            'percentil25': q25,
            'percentil75': q75,
            'error_mediana': error_median,
            'error_q25': error_q25,
            'error_q75': error_q75,
            'minimo': minimo,
            'maximo': maximo}
        return est
    
    def getID(self):
        return self.id
    
    def __str__(self):
        return f"Hoja {self.id}: areaAcum({self.getarea()}),apariciones({len(self.apariciones)})"
    
    
def xycentro(hoja, tam):
    return hoja.apariciones[tam-1].getx(), hoja.apariciones[tam-1].gety()

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

def es_duplicado(xmed, ymed, hojas, distancia_minima=5):
    # Recorre las hojas existentes y verifica la distancia
    for hoja in hojas:
        tam = hoja.getcantapariciones()
        xc, yc = hoja.apariciones[tam - 1].getx(), hoja.apariciones[tam - 1].gety()
        if math.hypot(xc - xmed, yc - ymed) < distancia_minima:
            return True
    return False

def comparar(dx, dy, xmed, ymed, area, frameactual, gv,kf): # Compara la posicion actual de la hoja con la posicion anterior para reconocer si pertenece o no a una anterior detección
    """
    Compara la posición actual de la hoja con las posiciones previas utilizando:
    - Distancia euclidiana.

    dx, dy: Ancho y altura del bounding box de la detección actual.
    xmed, ymed: Coordenadas del centro del bounding box de la detección actual.
    area: Área de la detección actual.
    frameactual: Frame actual del video.
    prev_gray, curr_gray: Frames anteriores y actuales en escala de grises.
    gv: Variable que almacena las hojas detectadas y sus atributos.
    kf: Lista que contiene los filtros de Kalman para cada hoja.
    """ 
    # Usar el factor k de la configuración si está disponible, sino usar 1.0 por defecto
    k = gv.configuracion.getfactork() if hasattr(gv, 'configuracion') and gv.configuracion else 1.0
    hp=math.hypot(dx, dy)
    centro=(xmed, ymed)
    color = (255, 0, 0)
    
    cv2.circle(gv.annotated_frame, centro, int(hp), color, 2) # Prueba de distancia 
    
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
            ultimo_frame = hoja.apariciones[tam-1].getframe()
            diferencia_frames = frameactual - ultimo_frame
            if diferencia_frames>0:
                if (-k*hp)<xc-xmed<(k*hp) and (-k*hp)<yc-ymed<(k*hp):
                    gv.ID=hoja.getID()
                    xp, yp= kf[gv.ID].predict(xmed, ymed)
                    if es_duplicado(xmed, ymed, gv.hojas, distancia_minima=0):
                        hoja.apariciones[tam-1].incrementar_bandera()
                    apar=Aparicion(xmed, ymed, xp, yp, hp, area, frameactual)
                    hoja.addaparicion(apar)

                    encontrado=1
                elif (-2*k*hp)<xc-xmed<(2*k*hp) and (-2*k*hp)<yc-ymed<(2*k*hp):
                    hoja.getID()
                    xp, yp= kf[gv.ID].predict(xmed, ymed)
                    if es_duplicado(xmed, ymed, gv.hojas, distancia_minima=0):
                        hoja.apariciones[tam-1].incrementar_bandera()
                    apar=Aparicion(xmed, ymed, xp, yp, hp, area, frameactual)
                    hoja.addaparicion(apar)
                    encontrado=1
        
        if not encontrado and (ymed > gv.yinicio or ymed < gv.yfinal) and not es_duplicado(xmed, ymed, gv.hojas, distancia_minima=5):
                gv.ID+=1
                kf.append(KalmanFilter())
                xp, yp= kf[gv.ID].predict(xmed, ymed)
                apar=Aparicion(xmed, ymed, xp, yp, hp, area, frameactual)
                gv.hojas.append(Hoja(apar, gv.ID))
    
    cv2.circle(gv.annotated_frame, centro, int(k*hp), color, 2)

