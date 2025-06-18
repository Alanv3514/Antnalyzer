import cv2
import numpy as np
import torch
torch.serialization.safe_load = True  # Establecer safe_load a True antes de cargar el modelo
import imutils
import math
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
from tkinter import *
from tkinter import filedialog as fd
from tkinter import messagebox as msg
from tkinter.messagebox import showinfo
from tkinter.ttk import Notebook, Frame
from time import sleep
from ultralytics import YOLO
from PIL import Image as ImgPIL
from PIL import ImageTk
from CTkToolTip import *
import pickle

from src.modules.hoja import Hoja, posicion, comparar, Aparicion, xycentro, xypredic
from src.modules.ToolTip import *
from src.modules.KalmanFilter import KalmanFilter
from src.modules.state_manager import StateManager

#Defino la variable global
class VGlobals:
    def __init__(self):
        self.PASSES=None
        self.paused = False
        self.point1 = None
        self.point2 = None
        self.bb = False
        self.show_debug_window = False  # Variable para controlar ventana de debug
        self.filtrado_estricto = False  # Variable para controlar tipo de filtrado
        
        # Estados de configuración
        self.area_seleccionada = False
        self.entrada_salida_seleccionada = False
        self.conversion_seleccionada = False
        
        # Puntos de conversión
        self.conv_p1 = None
        self.conv_p2 = None
        
        # Ruta del modelo
        self.model_path = "src/models_data/10-3.pt"
        
gv=VGlobals()


class Configuracion:    #Clase de configuracion
    def __init__(self, fecha, hora, fps, fpsdist, fpsapa, conf, cantapa, tiempo, device, factork):
        self.fecha = fecha
        self.hora = hora
        self.fps = fps
        self.fpsdist = fpsdist
        self.fpsapa = fpsapa
        self.conf = conf
        self.cantapa = cantapa
        self.tiempo = tiempo
        self.device = device
        self.factork = factork  

    def getfecha(self):
        return self.fecha
    
    def gethora(self):
        return self.hora
    
    def sethora(self, hora):
        self.hora = hora
    
    def getfps(self):
        return self.fps
    
    def getfpsdist(self):
        return self.fpsdist
    
    def getfpsapa(self):
        return self.fpsapa
    
    def getconf(self):
        return self.conf
    
    def getcantapa(self):
        return self.cantapa
    
    def gettiempo(self):
        return self.tiempo
    
    def getdevice(self):
        return self.device
    
    def getfactork(self):
        return self.factork
    
    def __str__(self):
        return f"Config: fecha:{self.fecha}, hora:{self.hora}, FPS:{self.fps},FPSDist:{self.fpsdist}, FPSAp:{self.fpsapa}, Confianza:{self.conf}, Cantidad de apariciones: {self.cantapa}, Intervalo de Guardado: {self.tiempo}, Procesar con: {self.device}"

    def __json__(self):
        return '["config": {"fecha":{self.fecha}, "hora":{self.hora}, "FPS":{self.fps},"FPSDist":{self.fpsdist}, "FPSAp":{self.fpsapa}, "Confianza":{self.conf}, "CantidadApariciones": {self.cantapa}, "DeltaTiempo": {self.tiempo}},"data":]'


def calculararea(hojas_final):
    area=0
    for hoja in hojas_final:
        area += hoja.getarea()
    return area


def seleccionar_carpeta(tab_instance):
    carpeta = fd.askdirectory()
    if carpeta:
        gv.carpeta_seleccionada = carpeta
        tab_instance.cambiar_estado()




def iniciar(UI2):
    global gv
    
    gv.filenames = fd.askopenfilename(multiple=True, title='Seleccione los videos')
    if not gv.filenames:  # Si el usuario cancela la selección
        return
    
    UI2.cambiarcolor(UI2.gettxt1(),"#009929")
    UI2.cambiarcolor(UI2.gettxt2(),"#ff0800")
        
    gv.filename = gv.filenames[0]
    gv.hojas.clear()
    device = gv.configuracion.device
    
    gv.archi1 = open(os.path.join(gv.carpeta_seleccionada, "datos-" + str(gv.configuracion.getfecha()) + ".txt"), "w+")
    gv.archi2 = open(os.path.join(gv.carpeta_seleccionada, "intervalo-" + str(gv.configuracion.getfecha()) + ".txt"), "w+")
    gv.archi2.write("CantHojas,Mediana,Percentil25,Percentil75,Minimo,Maximo,Media,AreaTotal,TSE,TCT,Fecha,HoraInicio,HoraFin\n")
    
    gv.ID=-1
    # Elegimos el modelo de detección
    gv.model = YOLO(gv.model_path)
    gv.model.to(device)

    gv.cap = cv2.VideoCapture(gv.filename)
    print(gv.configuracion)
    
    # Inicialización del video y botones
    gv.frameactual = 0
    gv.frameaux = 0
    
    # Inicializar variables para procesamiento
    gv.entrada_coord = None
    gv.salida_coord = None
    gv.direccion = 'arriba_a_abajo'  # Valor por defecto
    
    # Habilitar solo los botones de selección desde el inicio
    UI2.getPausa().configure(state="disabled", fg_color="#2B2B2B", text="Play")
    UI2.getBaseBlanca().configure(state="normal", fg_color="#e5be01")
    UI2.getBotonSeleccionES().configure(state="normal", fg_color="#2B2B2B")
    UI2.getBotonConv().configure(state="normal", fg_color="#2B2B2B")
    
    # Restablecemos los estados de configuración
    gv.area_seleccionada = False
    gv.entrada_salida_seleccionada = False
    gv.conversion_seleccionada = False
    

        
    UI2.cambiartexto(UI2.gettxt1(), "1. Seleccione video para comenzar [Boton Iniciar] ✓")
    UI2.cambiarcolor(UI2.gettxt1(), "#009929")
    
    UI2.cambiartexto(UI2.gettxt2(), "2. Seleccione Área de detección [Boton Área de Deteccion] ◍\n[Click y arrastre para dibujar un rectangulo] ")
    UI2.cambiarcolor(UI2.gettxt2(),  "#e5be01")

    UI2.cambiartexto(UI2.gettxt3(), "3. Seleccione Entrada y Salida [Boton ↑↓] ✘\n[1er click en entrada. 2do click en salida (boca de ingreso al nido)] ")
    UI2.cambiarcolor(UI2.gettxt3(),"#ff0800")

    UI2.cambiartexto(UI2.gettxt4(), "4. Seleccione el recuadro de referencia [Boton Conversion] ✘\n[dos clics, uno en cada esquina opuestas del cuadrado de referencia] ")
    UI2.cambiarcolor(UI2.gettxt4(), "#ff0800")

    # # Resetear textos de configuración
    # UI2.cambiartexto(UI2.gettxt1(), "1. Seleccione Área de detección [clic y arrastre]")
    # UI2.cambiartexto(UI2.gettxt2(), "2. Entrada y Salida\n[1er click en entrada. 2do click en salida (boca de ingreso al nido)]")
    # UI2.cambiartexto(UI2.gettxt3(), "3. Factor de conversión\n[dos clics, uno en cada esquina opuestas del cuadrado de conversion]")
    
    ## Mensajes iniciales actualizados
    #UI2.cambiartexto(UI2.gettxt4(), "Configure los 3 parámetros")
    
    # Leer y mostrar el primer frame antes de poner en pausa
    success, img = gv.cap.read()
    if success:
        # Guardar una copia del frame para las selecciones
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gv.img = img_rgb.copy()
        
        # Ajustar el widget al tamaño del video
        UI2.ajustar_widget_a_video()
        
        # Usar el tamaño original del video (sin redimensionar)
        height, width = img_rgb.shape[:2]
        
        # Crear imagen para la interfaz con tamaño original
        im = ImgPIL.fromarray(img_rgb)
        display_img = ctk.CTkImage(light_image=im, dark_image=im, size=(width, height))
        
        # Actualizar video en UI
        lblVideo = UI2.getlblVideo()
        lblVideo.configure(image=display_img)
        lblVideo.image = display_img
        
        # Activar botón del área (antes deshabilitado)
        UI2.getBaseBlanca().configure(state="normal")
    
    # Iniciar en pausa
    gv.paused = True

def on_pause(UI2):
    global gv
    
    # Verificar si todas las configuraciones están completas antes de cambiar el estado
    if not (gv.area_seleccionada and gv.entrada_salida_seleccionada and gv.conversion_seleccionada):
        # No permitir continuar si falta alguna configuración
        gv.paused = True
        UI2.getPausa().configure(text="Play", state="disabled", fg_color="#2B2B2B")
        return
    
    # Si todas las configuraciones están completas, proceder normalmente
    gv.paused = not gv.paused
    pausa = UI2.getPausa()
    
    if gv.paused:
        pausa.configure(text="Play", fg_color="#4E8F69")
    else:
        pausa.configure(text="Pausado", fg_color="#4E8F69")
            
    visualizar(UI2)

def detector(results, frameactual):
    if results[0].masks is None:
        pass
    else:
        indexaux=0
        index=0
        for result in results:   
            for i in range(len(result.masks.data)):
                tmp=(result.masks.data[i].cpu().numpy() * 255).astype("uint8")
                boxes=result.boxes[i].cpu().numpy()
                dx, dy, xmed, ymed =posicion(boxes)
                contorno, jerarquia = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contorno:
                    area=cv2.contourArea(c)
                comparar(dx, dy, xmed, ymed, area, frameactual, gv, kf)

def capturar():
    global gv
    if gv.cap is not None:
        success, img = gv.cap.read()
        if success:
            # Crear la carpeta screenshots si no existe
            if not os.path.exists("screenshots"):
                os.makedirs("screenshots")
            # Obtener el listado de archivos en la carpeta screenshots
            files = os.listdir("screenshots")
            # Si no existen capturas, crear la primera
            if not files:
                next_name = "screenshots/captura_1.png"
            else:
                # Buscar el último archivo creado
                print(files)
                last_file = files[-1]
                print(last_file)
                # Extraer el número del nombre del archivo
                last_num = int(last_file.split(".")[0].split("_")[1])
                print(last_num)
                # Generar el nombre de la siguiente captura
                next_num = last_num + 1
                next_name = f"screenshots/captura_{next_num}.png"
                print(next_name)
            # Guardar la captura con el nombre generado
            cv2.imwrite(next_name, img)
            
def base_blanca(UI2):  # Cuando apretamos el boton ponemos el flag up, para poder seleccionar la base
    global gv, seleccion_entrada_habilitada, seleccion_conversion
    
    gv.bb = True
    seleccion_entrada_habilitada = False
    seleccion_conversion = False
    gv.paused = True
    
    # Cambiar color de botones para indicar el modo activo
    UI2.getBaseBlanca().configure(fg_color="#f56767")  # Rojo para el botón activo
    
    UI2.cambiarcolor(UI2.gettxt1(),"#009929")

    # Respetar los colores de los botones según su estado de configuración
    if gv.entrada_salida_seleccionada:
        UI2.getBotonSeleccionES().configure(fg_color="#4E8F69")  # Verde si ya está configurado
        UI2.cambiarcolor(UI2.gettxt3(),"#009929")
    else:
        UI2.getBotonSeleccionES().configure(fg_color="#2B2B2B")  # Gris si no está configurado

        
    if gv.conversion_seleccionada:
        UI2.getBotonConv().configure(fg_color="#4E8F69")  # Verde si ya está configurado
        UI2.cambiarcolor(UI2.gettxt4(),"#009929")
    else:
        UI2.getBotonConv().configure(fg_color="#2B2B2B")  # Gris si no está configurado

    
    # Si ya estaba seleccionada, mantenemos el check
    if gv.area_seleccionada:
        UI2.cambiarcolor(UI2.gettxt2(),"#009929")
    
    visualizar(UI2)

def base_blanca_aux(point1, point2): # Aca nada mas cargamos las variables y cambia el texto
    global gv
    gv.y1 = point1[1]
    gv.x1 = point1[0]
    gv.y2 = point2[1]
    gv.x2 = point2[0]
    gv.yinicio = gv.y2 - 20
    gv.yfinal = gv.y1 + 20
    gv.area_seleccionada = True
    
def habilitar_seleccion(UI2):
    global gv, seleccion_entrada_habilitada, seleccion_conversion
    
    seleccion_entrada_habilitada = True
    seleccion_conversion = False
    gv.bb = False
    gv.paused = True
    
    # Cambiar color de botones para indicar el modo activo
    if gv.area_seleccionada:
        UI2.getBaseBlanca().configure(fg_color="#4E8F69")  # Verde si ya está configurado
        UI2.cambiarcolor(UI2.gettxt1(),"#009929")
    else:
        UI2.getBaseBlanca().configure(fg_color="#2B2B2B")  # Gris si no está configurado


    UI2.getBotonSeleccionES().configure(fg_color="#f56767")  # Rojo para el botón activo
    
    if gv.conversion_seleccionada:
        UI2.getBotonConv().configure(fg_color="#4E8F69")  # Verde si ya está configurado
        UI2.cambiarcolor(UI2.gettxt1(),"#009929")
    else:
        UI2.getBotonConv().configure(fg_color="#2B2B2B")  # Gris si no está configurado

    # Si ya estaba seleccionada, mantenemos el check
    if gv.entrada_salida_seleccionada:
        UI2.cambiarcolor(UI2.gettxt1(),"#009929")


    
    visualizar(UI2)

def habilitar_conversion(UI2):
    global gv, seleccion_conversion, seleccion_entrada_habilitada
    
    seleccion_conversion = True
    seleccion_entrada_habilitada = False
    gv.bb = False
    gv.paused = True
    
    # Cambiar color de botones para indicar el modo activo
    if gv.area_seleccionada:
        UI2.getBaseBlanca().configure(fg_color="#4E8F69")  # Verde si ya está configurado
        UI2.cambiarcolor(UI2.gettxt1(),"#009929")
    else:
        UI2.getBaseBlanca().configure(fg_color="#2B2B2B")  # Gris si no está configurado

        
    if gv.entrada_salida_seleccionada:
        UI2.getBotonSeleccionES().configure(fg_color="#4E8F69")  # Verde si ya está configurado
        UI2.cambiarcolor(UI2.gettxt2(),"#009929")
    else:
        UI2.getBotonSeleccionES().configure(fg_color="#2B2B2B")  # Gris si no está configurado

        
    UI2.getBotonConv().configure(fg_color="#f56767")  # Rojo para el botón activo
    
    # Si ya estaba seleccionada, mantenemos el check
    if gv.conversion_seleccionada:
        texto = UI2.gettxt3().cget("text").strip("\n")[0]+" ✓\n [%.2f mm²/px²]" % gv.cte
        UI2.cambiartexto(UI2.gettxt3(), texto)
        UI2.cambiarcolor(UI2.gettxt1(),"#009929")
    
    visualizar(UI2)

def eliminar_hojas(hojas, frame_actual):
    global gv # Hacer referencia a la variable global
    # Recorrer el array de hojas al revés para evitar problemas al eliminar elementos
    for i in range(len(hojas)-1, -1, -1):
        hoja = hojas[i]
        # Obtener la última aparición de la hoja
        primer_aparicion = hoja.apariciones[0]
        ultima_aparicion = hoja.apariciones[-1]
        bandera_superada = any(aparicion.match_count > 1 for aparicion in hoja.apariciones)
        # Si la distancia entre el frame de la última aparición y el frame actual es mayor a 15
        if frame_actual - ultima_aparicion.getframe() > gv.configuracion.getfpsapa():
            # Agregar la hoja al array de hojas perdidas
            if hoja.getcantapariciones()>gv.configuracion.getcantapa():
                gv.valid_ID += 1 # Aumentamos el ID valido
                if primer_aparicion.gety() > ultima_aparicion.gety() and bandera_superada==False:
                    hoja.valid_id = gv.valid_ID
                    gv.hojas_final.append(hoja)
                    if gv.filtrado_estricto:
                        gv.hojas_final = filtrar_duplicados_estricto(gv.hojas_final)
                    else:
                        gv.hojas_final = filtrar_duplicados(gv.hojas_final)
                elif bandera_superada==False:
                    hoja.valid_id = gv.valid_ID
                    gv.hojas_final_sale.append(hoja)
            # Eliminar la hoja del array hojas
            del hojas[i]
    return hojas
      
def detectar_direccion_entrada_salida(punto_entrada, punto_salida):
    dx = punto_salida[0] - punto_entrada[0]
    dy = punto_salida[1] - punto_entrada[1]

    if abs(dx) > abs(dy):
        if dx > 0:
            return 'izquierda_a_derecha'  # Horizontal de izquierda a derecha
        else:
            return 'derecha_a_izquierda'  # Horizontal de derecha a izquierda
    else:
        if dy > 0:
            return 'arriba_a_abajo'  # Vertical de arriba a abajo
        else:
            return 'abajo_a_arriba'  # Vertical de abajo a arriba

def rotar_imagen(imagen, direccion):    # Rotamos la imagen dependiendo la direccion de entrada->salida
    global gv
    if direccion == 'derecha_a_izquierda':
        imagen_rotada = cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE)
        gv.yinicio=imagen_rotada.shape[0] - 20
        gv.yfinal= 0 + 20
    elif direccion == 'izquierda_a_derecha':
        imagen_rotada = cv2.rotate(imagen, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gv.yinicio=imagen_rotada.shape[0] - 20
        gv.yfinal= 0 + 20
    elif direccion == 'arriba_a_abajo':
        imagen_rotada = cv2.rotate(imagen, cv2.ROTATE_180)
        gv.yinicio=imagen_rotada.shape[0] - 20
        gv.yfinal= 0 + 20
    else:  # 'abajo_a_arriba'
        imagen_rotada = imagen  # Ya está en la orientación deseada
        gv.yinicio=imagen_rotada.shape[0] - 20
        gv.yfinal= 0 + 20
    return imagen_rotada

def calcular_TSE(hojas_final):
    """
    Calcula la Tasa de Seguimiento Exitoso.
    
    Args:
        hojas_final (list): Lista de hojas con trayectorias completas
    Returns:
        float: Tasa de seguimiento exitoso
    """

    if not hasattr(gv, 'id_anterior'):
        gv.id_anterior = 0

    valid_ids_in_interval = gv.valid_ID - gv.id_anterior
    tse = len(hojas_final) / valid_ids_in_interval if valid_ids_in_interval > 0 else 0.0

    gv.id_anterior = gv.valid_ID
    
    return tse

def calcular_TCT(hojas_final):
    """
    Calcula la Tasa de Completitud de Trayectoria.
    
    Esta métrica mide qué tan bien el algoritmo mantiene el seguimiento 
    considerando la frecuencia de detección configurada.
    
    Returns:
        float: Valor entre 0 y 1, donde 1 indica seguimiento perfecto
    """
    tasas_completitud = []
    
    for hoja in hojas_final:
        # Obtener primer y último frame
        primer_frame = hoja.apariciones[0].getframe()
        ultimo_frame = hoja.apariciones[-1].getframe()
        
        # Calcular cuántas detecciones deberíamos tener
        frames_totales = ultimo_frame - primer_frame
        detecciones_esperadas = frames_totales / gv.configuracion.getfpsdist()
        
        # Cuántas detecciones realmente tenemos
        detecciones_reales = len(hoja.apariciones)
        
        # Calcular tasa de completitud
        if detecciones_esperadas > 0:
            tasa = detecciones_reales / detecciones_esperadas
            # Limitar a 1 en caso de que tengamos más detecciones de las esperadas
            tasa = min(1.0, tasa)
            tasas_completitud.append(tasa)
    
    # Retornar el promedio de todas las tasas
    return np.mean(tasas_completitud) if tasas_completitud else 0.0

def evaluar_seguimiento(hojas_final):
    """
    Evalúa el rendimiento general del sistema de seguimiento.
    
    Args:
        hojas_final (list): Lista de hojas con trayectorias completas
    Returns:
        dict: Diccionario con las métricas calculadas
    """
    tse = calcular_TSE(hojas_final)
    tct = calcular_TCT(hojas_final)
    
    return {
        "Tasa_Seguimiento_Exitoso": tse,
        "Tasa_de_Trayectorias": tct
    }

def filtrar_duplicados(hojas):
    """
    Filtra las hojas duplicadas manteniendo la hoja original y eliminando solo los duplicados.
    Utiliza criterios más robustos para la detección de duplicados.
    """
    hojas_filtradas = []
    hojas_a_descartar = set()
    
    # Ordenar hojas por ID para mantener consistencia
    hojas_ordenadas = sorted(hojas, key=lambda x: x.getID())
    
    for i, hoja1 in enumerate(hojas_ordenadas):
        if hoja1.getID() in hojas_a_descartar:
            continue
            
        # Obtener todas las apariciones de la hoja1
        apariciones1 = hoja1.apariciones
        
        for j, hoja2 in enumerate(hojas_ordenadas[i+1:], i+1):
            if hoja2.getID() in hojas_a_descartar:
                continue
                
            # Obtener todas las apariciones de la hoja2
            apariciones2 = hoja2.apariciones
            
            # Verificar si hay superposición significativa en las trayectorias
            duplicado = False
            coincidencias = 0
            total_comparaciones = 0
            
            # Comparar las últimas N apariciones de cada hoja
            n_comparaciones = min(5, len(apariciones1), len(apariciones2))
            
            for k in range(n_comparaciones):
                ap1 = apariciones1[-(k+1)]
                ap2 = apariciones2[-(k+1)]
                
                # Verificar frames cercanos
                if abs(ap1.getframe() - ap2.getframe()) <= 5:
                    # Calcular distancia euclidiana
                    dist = math.hypot(ap1.getx() - ap2.getx(), 
                                    ap1.gety() - ap2.gety())
                    
                    # Verificar área similar
                    area_diff = abs(ap1.getarea() - ap2.getarea()) / max(ap1.getarea(), ap2.getarea())
                    
                    # Criterios más estrictos para considerar duplicado
                    if dist < 5 and area_diff < 0.3:  # 30% de diferencia máxima en área
                        coincidencias += 1
                    total_comparaciones += 1
            
            # Si hay suficientes coincidencias, considerar como duplicado
            if total_comparaciones > 0 and coincidencias / total_comparaciones > 0.6:  # 60% de coincidencias
                duplicado = True
                
                # Mantener la hoja con más apariciones o la que apareció primero
                if len(apariciones1) >= len(apariciones2):
                    hojas_a_descartar.add(hoja2.getID())
                else:
                    hojas_a_descartar.add(hoja1.getID())
                    break  # Salir del bucle interno ya que esta hoja será descartada
    
    # Filtrar las hojas manteniendo solo las no duplicadas
    hojas_filtradas = [hoja for hoja in hojas_ordenadas if hoja.getID() not in hojas_a_descartar]
    
    return hojas_filtradas

def filtrar_duplicados_estricto(hojas):
    """
    Filtra las hojas duplicadas incluyendo aquellas que tienen patrones de duplicación
    """
    hojas_filtradas = []
    hojas_a_descartar = set()

    for i, hoja1 in enumerate(hojas):
        if hoja1.getID() in hojas_a_descartar:
            continue

        duplicados_encontrados = 0  # Contador de duplicados para esta hoja

        for j, hoja2 in enumerate(hojas[i+1:], i+1):
            ult_apar1 = hoja1.apariciones[-1]
            ult_apar2 = hoja2.apariciones[-1]

            # Verificamos frames cercanos
            if abs(ult_apar1.getframe() - ult_apar2.getframe()) <= 5:
                dist = math.hypot(ult_apar1.getx() - ult_apar2.getx(), 
                                ult_apar1.gety() - ult_apar2.gety())
                if dist < 5:
                    duplicados_encontrados += 1
                    hojas_a_descartar.add(hoja1.getID())
                    hojas_a_descartar.add(hoja2.getID())

        # Si la hoja tiene más de un duplicado, la marcamos para eliminar
        if duplicados_encontrados > 1:
            hojas_a_descartar.add(hoja1.getID())

    # Filtramos las hojas
    hojas_filtradas = [hoja for hoja in hojas if hoja.getID() not in hojas_a_descartar]
    
    return hojas_filtradas

def escribirarchivo(hojas_final, hojas_final_sale, bandera):
    """
    Escribe datos en los archivos de salida.
    
    Args:
        hojas_final: Lista de hojas procesadas
        hojas_final_sale: Lista de hojas salientes
        bandera: 0 para escribir detalles, 1 para escribir resumen de intervalo
    """
    global gv
    
    try:
        # Verificar que los archivos estén abiertos
        if (bandera == 0 and not hasattr(gv, 'archi1')) or (bandera == 1 and not hasattr(gv, 'archi2')):
            print(f"Error: Archivo {'archi1' if bandera == 0 else 'archi2'} no disponible")
            return
            
        # Escribir datos en archivos según la bandera
        if bandera == 0:
            for item in hojas_final:
                for aparicion in item.apariciones:
                    gv.archi1.write(str(item.valid_id)+"|"+str(aparicion.getx()) +"|"+ 
                                  str(aparicion.gety()) +"|"+ str(aparicion.getxp()) +"|"+ 
                                  str(aparicion.getyp()) +"|"+str(aparicion.getarea())+ "|"+ 
                                  str(aparicion.getframe())+"\n")
        
        elif bandera == 1:
            # Inicializar estadísticas
            gv.estadisticas = []
            
            # Usar copia local de garch para evitar problemas con actualizaciones concurrentes
            garch_actual = gv.garch
            tiempo_guardado = gv.configuracion.gettiempo()
            
            # Obtener estadísticas de las hojas en el intervalo de tiempo
            gv.estadisticas = [item.getarea() for item in hojas_final 
                              if (garch_actual - tiempo_guardado) < 
                              (item.apariciones[0].getframe() / (30 * 60)) < garch_actual]
            
            # Ir al final del archivo
            gv.archi2.seek(0, 2)
            
            # Calcular métricas de seguimiento incluso si no hay hojas
            metricas = evaluar_seguimiento(hojas_final)
            tse = metricas["Tasa_Seguimiento_Exitoso"]
            tct = metricas["Tasa_de_Trayectorias"]

            minutos_transcurridos = int(garch_actual)
            hora_inicial = gv.configuracion.gethora()
            hora_fin = hora_inicial + datetime.timedelta(minutes=minutos_transcurridos)
            hora_inicio = hora_fin - datetime.timedelta(minutes=tiempo_guardado)

            fecha_inicial_str = gv.configuracion.getfecha()

            # Convertir la fecha de string a objeto datetime
            try:
                # Asumiendo formato DD-MM-YYYY
                dia, mes, anio = map(int, fecha_inicial_str.split('-'))
                fecha_inicial = datetime.datetime(anio, mes, dia)
                
                # Crear datetime completos para inicio y fin (fecha + hora)
                dt_inicio = fecha_inicial + hora_inicio
                dt_fin = fecha_inicial + hora_fin
                
                # Formatear las fechas y horas
                fecha_str = dt_fin.strftime("%d-%m-%Y")
                hora_inicio_str = dt_inicio.strftime("%H:%M")
                hora_fin_str = dt_fin.strftime("%H:%M")
                
            except ValueError:
                # En caso de error en el formato de fecha, usar valores por defecto
                fecha_str = fecha_inicial_str
                hora_inicio_str = f"{hora_inicio.days * 24 + hora_inicio.seconds // 3600:02d}:{(hora_inicio.seconds // 60) % 60:02d}"
                hora_fin_str = f"{hora_fin.days * 24 + hora_fin.seconds // 3600:02d}:{(hora_fin.seconds // 60) % 60:02d}"

            # Si hay estadísticas, usar valores calculados, si no hay, usar 0
            if len(gv.estadisticas) > 0:
                area_mediana = np.mean([item['mediana'] for item in gv.estadisticas])*gv.cte
                area_percentil25 = np.mean([item['percentil25'] for item in gv.estadisticas])*gv.cte
                area_percentil75 = np.mean([item['percentil75'] for item in gv.estadisticas])*gv.cte
                area_maxima = np.max([item['maximo'] for item in gv.estadisticas])*gv.cte
                area_minima = np.min([item['minimo'] for item in gv.estadisticas])*gv.cte
                area_media = np.mean([item['media'] for item in gv.estadisticas])*gv.cte
                area_total = sum([item['media'] for item in gv.estadisticas])*gv.cte  # Suma de todas las áreas medias
                cant_hojas = len(gv.estadisticas)
            else:
                # Si no hay hojas en este intervalo, todos los valores son 0
                area_mediana = 0.0
                area_percentil25 = 0.0
                area_percentil75 = 0.0
                area_maxima = 0.0
                area_minima = 0.0
                area_media = 0.0
                area_total = 0.0
                cant_hojas = 0

            # Escribir en el archivo siempre, independientemente de si hay datos o no
            gv.archi2.write(f"{cant_hojas},{area_mediana:.2f},{area_percentil25:.2f},"
               f"{area_percentil75:.2f},{area_minima:.2f},{area_maxima:.2f},"
               f"{area_media:.2f},{area_total:.2f},"
               f"{tse:.2f},{tct:.2f}," 
               f"{fecha_str},{hora_inicio_str},{hora_fin_str}\n")
            
            # Asegurar que se escriba inmediatamente en el archivo
            gv.archi2.flush()
            
            if not hasattr(gv, 'total_hojas'):
                gv.total_hojas = 0
            gv.total_hojas += len(hojas_final)
            
            # Limpiar la lista de hojas
            gv.hojas_final.clear()
            
    except Exception as e:
        print(f"Error en escribirarchivo (bandera {bandera}): {str(e)}")
        import traceback
        traceback.print_exc()

def visualizar(UI2):
    """
    Función principal para visualizar y procesar el video
    Args:
        UI2: Objeto de la interfaz de usuario
    """
    if gv.cap is None:
        return

    if not gv.paused:
        try:
            # Lectura del frame
            success, img = gv.cap.read()
            if not success:
                handle_end_of_video(UI2, gv)
                return

            # Incrementar contador de frames
            gv.frameactual += 1

            # Procesar frame solo en el área seleccionada
            frame = img[gv.y1:gv.y2, gv.x1:gv.x2]
            frame = rotar_imagen(frame, gv.direccion)

            # Procesar detecciones YOLO solo cuando sea necesario
            if gv.frameactual - gv.frameaux >= gv.configuracion.getfpsdist():
                try:
                    # Verificar que el frame no esté vacío
                    if frame is None or frame.size == 0:
                        print("Warning: Frame vacío o nulo en YOLO")
                        return
                        
                    # Usar la confianza de configuración para reducir falsos positivos
                    results = gv.model.predict(frame, conf=gv.configuracion.getconf(), verbose=False)
                    
                    # Verificar que results contenga datos válidos
                    if results and len(results) > 0:
                        gv.annotated_frame = results[0].plot()
                        detector(results, gv.frameactual)
                        
                        
                        if gv.show_debug_window:
                            try:
                                debug_frame = gv.annotated_frame.copy()
                                
                                # Redimensionar para la ventana de debug
                                debug_height, debug_width = debug_frame.shape[:2]
                                debug_ratio = 320 / debug_width
                                debug_new_height = int(debug_height * debug_ratio)
                                debug_resized = cv2.resize(debug_frame, (320, debug_new_height), interpolation=cv2.INTER_AREA)
                                
                                # Convertir para mostrar en la interfaz
                                debug_rgb = cv2.cvtColor(debug_resized, cv2.COLOR_BGR2RGB) if debug_resized.shape[2] == 3 else debug_resized
                                debug_im = ImgPIL.fromarray(debug_rgb)
                                debug_img = ctk.CTkImage(light_image=debug_im, dark_image=debug_im, size=(320, debug_new_height))
                                
                                # Actualizar la ventana de debug
                                UI2.debug_label.configure(image=debug_img)
                                UI2.debug_label.image = debug_img
                            except Exception as e:
                                print(f"Error al actualizar ventana de debug: {str(e)}")
                    else:
                        print("Warning: Resultados de YOLO vacíos")
                        
                    gv.frameaux = gv.frameactual
                except Exception as e:
                    print(f"Error en procesamiento YOLO: {e}")
                    import traceback
                    traceback.print_exc()

            # Cálculos de tiempo
            sec = gv.frameactual / gv.configuracion.getfps()
            s = datetime.timedelta(seconds=int(sec))
            hora = gv.configuracion.gethora() + s
            
            # Actualizar UI de manera eficiente
            UI2.cambiartexto(UI2.gettxt5(), str(hora))
            UI2.cambiartexto(UI2.gettxt4(), f"Hojas entrantes: {gv.total_hojas + len(gv.hojas_final)}")

            # Manejo de guardado automático
            gv.garch = sec / 60
            tiempo_guardado = float(gv.configuracion.gettiempo())
            
            # Calcular si toca guardar datos en archivos, con una pequeña tolerancia para redondeo
            guardar_archivos = tiempo_guardado > 0 and abs(gv.garch % tiempo_guardado) == 0
            

            # Guardar archivos de datos si corresponde
            if guardar_archivos:
                try:
                    # Escribir en los archivos de datos
                    escribirarchivo(gv.hojas_final, gv.hojas_final_sale, 0)
                    escribirarchivo(gv.hojas_final, gv.hojas_final_sale, 1)
                    filepath = "antnalyzer_state.pkl"  # Siempre en la carpeta base
                    UI2.getlblVideo().after_idle(lambda: StateManager.save_application_state(gv, filepath))
                except Exception as e:
                    print(f"Error al escribir en archivos: {str(e)}")
            

            # Procesar hojas
            gv.hojas = eliminar_hojas(gv.hojas, gv.frameactual)

            # Dibujar elementos visuales solo en la imagen original
            cv2.rectangle(img, (gv.x1, gv.y1), (gv.x2, gv.y2), (0, 255, 0), 2)
            
            # Dibujar puntos de entrada/salida solo si están definidos
            if gv.entrada_coord:
                cv2.circle(img, gv.entrada_coord, radius=5, color=(0, 255, 0), thickness=-1)
            if gv.salida_coord:
                cv2.circle(img, gv.salida_coord, radius=5, color=(0, 0, 255), thickness=-1)

            # Convertir y guardar una copia del frame original
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gv.img = img_rgb.copy()
            
            # Obtener referencia al widget de video
            lblVideo = UI2.getlblVideo()
            
            # Usar tamaño original del video
            height, width = img_rgb.shape[:2]
            
            # Crear imagen para la interfaz con tamaño original
            im = ImgPIL.fromarray(img_rgb)
            display_img = ctk.CTkImage(light_image=im, dark_image=im, size=(width, height))
            
            # Actualizar video en UI
            lblVideo.configure(image=display_img)
            lblVideo.image = display_img

            # Actualizar barra de progreso
            if gv.frameactual % 5 == 0:  # Actualizar cada 5 frames para reducir carga
                current_frame = gv.cap.get(cv2.CAP_PROP_POS_FRAMES)
                total_frames = gv.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                progress = current_frame / total_frames
                UI2.getProgressBar().set(progress)
                
                
            # Programar siguiente frame con un intervalo adecuado para mantener fluidez
            lblVideo.after(1, lambda: visualizar(UI2))

        except Exception as e:
            print(f"Error en visualizar: {e}")
            
            # Intentar guardar el estado antes de cerrar en caso de error
            try:
                filepath = "antnalyzer_state.pkl"  # Siempre en la carpeta base
                StateManager.save_application_state(gv, filepath)
            except:
                pass
                
            if gv.cap is not None:
                gv.cap.release()
            cv2.destroyAllWindows()

def handle_end_of_video(UI2, gv):
    """
    Maneja el final del video actual
    """
    try:
        # Liberar recursos actuales
        if hasattr(gv, 'cap') and gv.cap is not None:
            gv.cap.release()
        
        # Verificar si hay una lista de archivos y un archivo actual válidos
        if not hasattr(gv, 'filenames') or not gv.filenames or not hasattr(gv, 'filename') or not gv.filename:
            print("No hay archivos de video para procesar o ya se terminaron todos.")
            finalizar_procesamiento(UI2)
            return
        
        # Determinar si estamos en el último
        try:
            current_index = gv.filenames.index(gv.filename)
            is_last_video = current_index >= len(gv.filenames) - 1
        except ValueError:
            # Si no se encuentra el archivo en la lista, asumimos que es el último
            print("Advertencia: No se pudo determinar la posición del archivo actual en la lista.")
            is_last_video = True
        
        if is_last_video:
            # Es el último video, finalizamos todo
            finalizar_procesamiento(UI2)
        else:
            # Pasar al siguiente video
            next_index = current_index + 1
            gv.filename = gv.filenames[next_index]
            
            try:
                gv.cap = cv2.VideoCapture(gv.filename)
                if not gv.cap.isOpened():
                    print(f"Error: No se pudo abrir el video siguiente: {gv.filename}")
                    # Intentar con el siguiente si este falla
                    handle_end_of_video(UI2, gv)
                    return
                # Continuar con la visualización
                visualizar(UI2)
            except Exception as e:
                print(f"Error al abrir el siguiente video: {str(e)}")
                finalizar_procesamiento(UI2)
    except Exception as e:
        print(f"Error al manejar el final del video: {str(e)}")
        import traceback
        traceback.print_exc()
        finalizar_procesamiento(UI2)

def finalizar_procesamiento(UI2):
    """
    Finaliza el procesamiento de todos los videos, cerrando archivos y liberando recursos
    """
    try:
        # Cerrar archivos de salida
        if hasattr(gv, 'archi1') and gv.archi1:
            try:
                gv.archi1.close()
            except:
                pass
                
        if hasattr(gv, 'archi2') and gv.archi2:
            try:
                gv.archi2.close()
            except:
                pass
        
        # Cerrar todas las ventanas de OpenCV
        cv2.destroyAllWindows()
        
        # Eliminar el archivo de estado al completar todo el procesamiento
        try:
            filepath = "antnalyzer_state.pkl"  # Siempre en la carpeta base
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Archivo de estado eliminado: {filepath}")
                
                # Mostrar mensaje de finalización
                msg.showinfo("Procesamiento Completado", "El procesamiento de todos los videos ha finalizado correctamente.")
        except Exception as e:
            print(f"Error al eliminar archivo de estado: {str(e)}")
            
        # Actualizar la interfaz para indicar que se ha completado
        if UI2:
            UI2.cambiartexto(UI2.gettxt4(), "Procesamiento completado")
            UI2.getPausa().configure(state="disabled")
            
    except Exception as e:
        print(f"Error al finalizar el procesamiento: {str(e)}")


# Variables

gv.font=cv2.FONT_HERSHEY_SIMPLEX 
gv.total_hojas=0
gv.frameactual=0
gv.frameaux=0
gv.cte=0
gv.bb=False
kf=[]
kf.append(KalmanFilter())
yfinalaux= 240
gv.ID=-1
gv.valid_ID=0
gv.x1=0
gv.x2=640
gv.y1=0
gv.y2=480
gv.yinicio=gv.y2 - 20
gv.yfinal= gv.y1 + 20
gv.hojas=[]
gv.hojas_final=[]
gv.hojas_final_sale=[]
click_count = 0
gv.point1 = None
gv.point2 = None
gv.paused = False
primera = False
gv.entrada_coord = None
gv.salida_coord = None
gv.direccion= None
seleccion_entrada_habilitada = False
seleccion_conversion = False




# Clase principal de la aplicación
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración de la ventana principal
        self.title("GUI | CUSTOMTKINTER | HOJAS")
        self.geometry("1280x720")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("assets/marsh.json")
        
        # Configuración de las pestañas
        self.pestanias = MyTabView(master=self)
        self.pestanias.pack(expand=True, fill='both')

        # Cargar estado si existe
        self.state_loaded = self.load_state()
        
        # Agregar protocolo de cierre
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_state(self):
        """Intenta cargar el estado guardado de la aplicación"""
        global gv
        filepath = "antnalyzer_state.pkl"  # Buscar siempre en la carpeta base
        
        # Verificar si existe un archivo de estado
        if os.path.exists(filepath):
            try:
                # Obtener información básica del archivo de estado para mostrarla
                with open(filepath, 'rb') as f:
                    state_dict = pickle.load(f)
                
                # Obtener datos relevantes para mostrar
                timestamp = state_dict.get('timestamp', 'desconocido')
                filename = state_dict.get('filename', 'ninguno') if state_dict.get('filename') else 'ninguno'
                filename = os.path.basename(filename) if filename != 'ninguno' else 'ninguno'

                
                # Construir mensaje informativo
                mensaje = (f"Se encontró una sesión guardada:\n\n"
                          f"• Fecha: {timestamp}\n"
                          f"• Video: {filename}\n\n"
                          "¿Desea cargar esta sesión y continuar desde donde se quedó?")
                
                # Mostrar cuadro de diálogo para confirmar
                respuesta = msg.askyesno("Sesión Anterior Encontrada", mensaje)
                
                if respuesta:
                    # El usuario quiere cargar el estado
                    loaded = StateManager.load_application_state(gv, filepath)
                    
                    if loaded:
                        # Si se cargó el estado correctamente, configurar la interfaz
                        self.after(500, self.restore_ui_state)
                        return True
                else:
                    # El usuario no quiere cargar el estado, eliminar el archivo
                    try:
                        os.remove(filepath)
                        print(f"Archivo de estado eliminado: {filepath}")
                    except Exception as e:
                        print(f"Error al eliminar archivo de estado: {str(e)}")
            except Exception as e:
                print(f"Error al obtener información del estado: {str(e)}")
                
                # Preguntar de todas formas si hay un error al leer
                respuesta = msg.askyesno("Sesión Anterior Encontrada", 
                                       "Se encontró una sesión guardada, pero no se pudo leer su información.\n"
                                       "¿Desea intentar cargar esta sesión?")
                
                if respuesta:
                    # El usuario quiere cargar el estado a pesar del error
                    loaded = StateManager.load_application_state(gv, filepath)
                    
                    if loaded:
                        # Si se cargó el estado correctamente, configurar la interfaz
                        self.after(500, self.restore_ui_state)
                        return True
                else:
                    # El usuario no quiere cargar el estado, eliminar el archivo
                    try:
                        os.remove(filepath)
                        print(f"Archivo de estado eliminado: {filepath}")
                    except Exception as e:
                        print(f"Error al eliminar archivo de estado: {str(e)}")
        
        return False
    
    def restore_ui_state(self):
        """Restaura el estado de la interfaz de usuario después de cargar un estado guardado"""
        global gv, kf
        
        try:
            # Si no hay configuración cargada, no podemos restaurar
            if not hasattr(gv, 'configuracion') or gv.configuracion is None:
                print("No hay configuración cargada para restaurar el estado")
                return
            
            # Ir a la tab de Pantalla Video
            self.pestanias.siguiente("Pantalla Video")
            
            # Obtener la instancia de Tab2
            tab2_instance = None
            for child in self.pestanias.tab("Pantalla Video").winfo_children():
                if isinstance(child, Tab2):
                    tab2_instance = child
                    break

            
            if tab2_instance:
                # Inicializar filtros Kalman
                kf = [KalmanFilter()]  # Reiniciar lista de filtros Kalman
                
                # Asegurarse de que ID está sincronizado con el número de filtros Kalman
                if hasattr(gv, 'ID') and gv.ID >= 0:
                    # Para cada ID adicional, crear un nuevo filtro Kalman
                    for i in range(gv.ID):
                        kf.append(KalmanFilter())
                
                # Reabrir el video y posicionar en el frame guardado
                if hasattr(gv, 'filename') and gv.filename:
                    # Si no hay una captura abierta, abrirla
                    if not hasattr(gv, 'cap') or gv.cap is None:
                        # Reabrir archivos de salida en modo append
                        try:
                            fecha = gv.configuracion.getfecha()
                            carpeta = gv.carpeta_seleccionada
                            
                            gv.archi1 = open(os.path.join(carpeta, f"datos-{fecha}.txt"), "a+")
                            gv.archi2 = open(os.path.join(carpeta, f"intervalo-{fecha}.txt"), "a+")
                            
                            # Verificar si el archivo está vacío, agregar encabezado si es necesario
                            gv.archi2.seek(0, 0)
                            if not gv.archi2.read(1):
                                gv.archi2.write("CantHojas,Mediana,Percentil25,Percentil75,Minimo,Maximo,Media,AreaTotal,TSE,TCT,Fecha,HoraInicio,HoraFin\n")
                            
                            # Volver al final del archivo
                            gv.archi2.seek(0, 2)
                            
                        except Exception as e:
                            print(f"Error al abrir archivos de salida: {str(e)}")
                        
                        # Inicializar variables críticas si no existen
                        if not hasattr(gv, 'annotated_frame'):
                            gv.annotated_frame = None
                            
                        if not hasattr(gv, 'hojas'):
                            gv.hojas = []
                            
                        if not hasattr(gv, 'hojas_final'):
                            gv.hojas_final = []
                            
                        if not hasattr(gv, 'hojas_final_sale'):
                            gv.hojas_final_sale = []
                        
                        # Cargar el modelo YOLO
                        if hasattr(gv, 'model_path'):
                            try:
                                device = gv.configuracion.device
                                gv.model = YOLO(gv.model_path)
                                gv.model.to(device)
                            except Exception as e:
                                print(f"Error al cargar modelo YOLO: {str(e)}")
                                return  # No continuar si no se puede cargar el modelo
                        else:
                            print("No hay ruta del modelo para restaurar")
                            return
                        
                        # Abrir el video
                        try:
                            gv.cap = cv2.VideoCapture(gv.filename)
                            if not gv.cap.isOpened():
                                print(f"No se pudo abrir el video: {gv.filename}")
                                return
                                
                            # Posicionar en el frame guardado
                            if gv.frameactual > 0:
                                # Asegurarse de que no estamos intentando ir más allá del final del video
                                total_frames = int(gv.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                if gv.frameactual >= total_frames:
                                    gv.frameactual = max(0, total_frames - 1)
                                
                                gv.cap.set(cv2.CAP_PROP_POS_FRAMES, gv.frameactual)
                                
                            # Leer el primer frame para asegurarse de que todo está bien
                            success, img = gv.cap.read()
                            if not success:
                                print("No se pudo leer el primer frame del video restaurado")
                                return
                                
                            # Guardar una copia del frame
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            gv.img = img_rgb.copy()
                            
                        except Exception as e:
                            print(f"Error al abrir el video: {str(e)}")
                            return
                    
                    # Actualizar interfaz según el estado de configuración
                    self.update_ui_from_state(tab2_instance)
                    
                    # Para evitar problemas con gv.paused
                    if not hasattr(gv, 'paused'):
                        gv.paused = True
                    
                    if not gv.paused:
                        tab2_instance.getPausa().configure(text="Pausa", fg_color="#4E8F69")
                        # Usar after para dar tiempo a que todo se inicialice correctamente
                        tab2_instance.getlblVideo().after(1000, lambda: visualizar(tab2_instance))
        
        except Exception as e:
            print(f"Error al restaurar estado de UI: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_ui_from_state(self, tab2_instance):
        """Actualiza la interfaz de usuario basado en el estado cargado"""
        global gv
        
        # Actualizar textos de configuración según el estado
        if gv.area_seleccionada:
            tab2_instance.cambiartexto(tab2_instance.gettxt1(), "1. Área de detección [Boton Área de Deteccion] ✓")
            tab2_instance.cambiarcolor(tab2_instance.gettxt1(),"#009929")
            tab2_instance.getBaseBlanca().configure(fg_color="#4E8F69")


        if gv.entrada_salida_seleccionada:
            tab2_instance.cambiartexto(tab2_instance.gettxt2(), "2. Entrada y Salida [Boton ↑↓] ✓")
            tab2_instance.cambiarcolor(tab2_instance.gettxt2(),"#009929")
            tab2_instance.getBaseBlanca().configure(fg_color="#4E8F69")
            
        
        if gv.conversion_seleccionada:
            texto = f"4. Conversion ✓ [{gv.cte:.2f} mm²/px²]" if hasattr(gv, 'cte') else "4. Conversion ✓"
            tab2_instance.cambiartexto(tab2_instance.gettxt3(), texto)
            tab2_instance.cambiarcolor(tab2_instance.gettxt3(),"#009929")
            tab2_instance.getBotonConv().configure(fg_color="#4E8F69")
            
            # Actualizar también el nuevo widget de conversión
            if hasattr(gv, 'cte'):
                texto_conversion = f"Factor de Conversión: {gv.cte:.2f} mm²/px²"
                tab2_instance.cambiartexto(tab2_instance.getTextoConversion(), texto_conversion)
                tab2_instance.cambiarcolor(tab2_instance.getTextoConversion(), "#4E8F69")
        
        # Si todas las configuraciones están completas, habilitar el botón de pausa
        if gv.area_seleccionada and gv.entrada_salida_seleccionada and gv.conversion_seleccionada:
            tab2_instance.cambiartexto(tab2_instance.gettxt4(), "Configuración completa. Pulse Play")
            tab2_instance.getPausa().configure(state="normal", fg_color="#4E8F69")
        
        # Actualizar contador de hojas
        if hasattr(gv, 'total_hojas'):
            tab2_instance.cambiartexto(tab2_instance.gettxt4(), f"Hojas entrantes: {gv.total_hojas + len(gv.hojas_final)}")
        
        # Actualizar selector de filtrado
        if hasattr(gv, 'filtrado_estricto'):
            valor_filtrado = "Estricto" if gv.filtrado_estricto else "Normal"
            tab2_instance.filtrado_var.set(valor_filtrado)
    
    def save_state(self):
        """Guarda el estado actual de la aplicación"""
        global gv
        
        # Solo guardar si hay un video en procesamiento
        if hasattr(gv, 'filename') and gv.filename and hasattr(gv, 'frameactual') and gv.frameactual > 0:
            filepath = "antnalyzer_state.pkl"  # Guardar siempre en la carpeta base
            StateManager.save_application_state(gv, filepath)
    
    def on_closing(self):
        """Maneja el cierre de la aplicación"""
        try:
            
            # Limpiar recursos de matplotlib
            tab3 = self.pestanias.tab("Análisis").winfo_children()[0]
            if isinstance(tab3, Tab3):
                tab3.cleanup()
                
            # Cerrar la aplicación
            self.quit()
            self.destroy()
        except:
            self.quit()
            self.destroy()



class MyTabView(ctk.CTkTabview):
    def __init__(self, master, **kwargs):
        super().__init__(master, anchor="w", **kwargs)

        #Creamos las pestañas
        self.add("Init")
        self.add("Pantalla Video")
        self.add("Análisis")

        self.tab("Init").configure(border_width=0)  
        Tab1(self.tab("Init"), parent=self)

        self.tab("Pantalla Video").configure(border_width=0)  
        Tab2(self.tab("Pantalla Video"))

        self.tab("Análisis").configure(border_width=0)
        Tab3(self.tab("Análisis"))  
        

    def habilitar_tabs(self):
        self.configure(state="normal")

    def siguiente(self, nombre_pest):
        self.set(nombre_pest)

# Clase para la primera pestaña
class Tab1(ctk.CTkFrame):
    def __init__(self, master, parent=None):
        super().__init__(master)
        # Configuración de los widgets de la pestaña 1
        self.parent=parent
        
        # Configurar el grid para permitir espacio para los logos
        self.grid_columnconfigure(4, weight=1)  # Columna para los logos
        imagenQ = ctk.CTkImage(light_image=ImgPIL.open('assets/interrogatorio_2.png'),
                                     dark_image=ImgPIL.open('assets/interrogatorio_2.png'),
                                     size=(16,16))
        
        self.fechastring = tk.StringVar(value="01-01-1970")
        self.horastring = ctk.StringVar(value="00:00")
        self.fpstring = ctk.IntVar(value=30)
        self.fpsdisstring = ctk.IntVar(value=2)
        self.fpsapastring = ctk.IntVar(value=15)
        self.confstring = ctk.DoubleVar(value=0.6)
        self.cantstring = ctk.IntVar(value=10)
        self.tiemstring = ctk.DoubleVar(value=10)
        self.factork_string = ctk.DoubleVar(value=1.0)

# Configuración de dispositivos
        self.devices = ['cpu']  # Siempre incluir CPU
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.devices.append(f"cuda:{i} ({torch.cuda.get_device_name(i)})")
        
        # Variable para almacenar el dispositivo seleccionado
        self.device_var = ctk.StringVar(value=self.devices[0])

        # Funciones de validación
        self.vcmd_int = (self.register(lambda P: self.callback('int', P)), '%P')
        self.vcmd_float = (self.register(lambda P: self.callback('float', P)), '%P')
        
        
        #Widgets de fecha
        fechat = ctk.CTkLabel(self, text="Fecha del vídeo:")
        fechat.grid(row=0, column=0, sticky="w")
        fecha = ctk.CTkEntry(self, textvariable=self.fechastring, width=150)
        fecha.grid(row=0, column=1, sticky="w")
        fechaq = ctk.CTkButton(self, fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        fechaq.grid(row=0, column=2, sticky="w", padx=5)

        self.crear_toolTip(fechaq, 'Fecha del video en formato DD-MM-YYYY')
        
        #Widget de hora
        horat = ctk.CTkLabel(self, text="Hora de inicio:").grid(row=1, column=0, sticky=W)
        hora = ctk.CTkEntry(self, textvariable = self.horastring, width=150).grid(row=1, column=1, sticky=W)
        horaq = ctk.CTkButton(self,fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        horaq.grid(row=1, column=2, sticky=W, padx=5)
        self.crear_toolTip(horaq, 'Hora de inicio del video en formato HH:MM')

        fpst = ctk.CTkLabel(self, text="Fotogramas por Segundo:").grid(row=2, column=0, sticky="w")
        FPS = ctk.CTkEntry(self, textvariable=self.fpstring, width=150, validatecommand=self.vcmd_int)
        FPS.grid(row=2, column=1, sticky="w")
        fpsq = ctk.CTkButton(self, fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        fpsq.grid(row=2, column=2, sticky="w", padx=5)
        self.crear_toolTip(fpsq, 'Fotogramas por Segundo en que fue grabado el vídeo')
        
        fpsdist = ctk.CTkLabel(self, text="Distancia de Frames:").grid(row=3, column=0, sticky="w")
        fpsdis = ctk.CTkEntry(self, textvariable=self.fpsdisstring, width=150, validate="key", validatecommand=self.vcmd_int)
        fpsdis.grid(row=3, column=1, sticky="w")
        fpsdisq = ctk.CTkButton(self, fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        fpsdisq.grid(row=3, column=2, sticky="w", padx=5)
        self.crear_toolTip(fpsdisq, 'Distancia entre frames de detección, cuanto mayor sea este numero\nmas rapido será el procesamiento a cambio de un mayor error')

        fpsapat = ctk.CTkLabel(self, text="Frames aparición:").grid(row=4, column=0, sticky="w")
        fpsapa = ctk.CTkEntry(self, textvariable=self.fpsapastring, width=150, validate="key", validatecommand=self.vcmd_int)
        fpsapa.grid(row=4, column=1, sticky="w")
        fpsapaq = ctk.CTkButton(self, fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        fpsapaq.grid(row=4, column=2, sticky="w", padx=5)
        self.crear_toolTip(fpsapaq, 'Cantidad de frames que deben pasar para dar por terminada una detección.\nSe recomienda no utilizar un valor mayor al de FPS')

        conft = ctk.CTkLabel(self, text="Confianza:").grid(row=5, column=0, sticky="w")
        conf = ctk.CTkEntry(self, textvariable=self.confstring, width=150, validate="key", validatecommand=self.vcmd_float)
        conf.grid(row=5, column=1, sticky="w")
        confq = ctk.CTkButton(self, fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        confq.grid(row=5, column=2, sticky="w", padx=5)
        self.crear_toolTip(confq, 'Valor de umbral de confianza del modelo, entre 0 y 1, cuanto mayor sea el valor habrá menos falso positivos,\npero se perderán detecciones')

        cantapat = ctk.CTkLabel(self, text="Cantidad de apariciones:").grid(row=6, column=0, sticky="w")
        cantapa = ctk.CTkEntry(self, textvariable=self.cantstring, width=150, validate="key", validatecommand=self.vcmd_int)
        cantapa.grid(row=6, column=1, sticky="w")
        cantapaq = ctk.CTkButton(self, fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        cantapaq.grid(row=6, column=2, sticky="w", padx=5)
        self.crear_toolTip(cantapaq, 'Cantidad de apariciones mínimas necesarias para dar por positiva la completa detección')

        tiemt = ctk.CTkLabel(self, text="Intervalo de Guardado:").grid(row=7, column=0, sticky="w")
        tiem = ctk.CTkEntry(self, textvariable=self.tiemstring, width=150, validate="key", validatecommand=self.vcmd_float)
        tiem.grid(row=7, column=1, sticky="w")
        tiemq = ctk.CTkButton(self, fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        tiemq.grid(row=7, column=2, sticky="w", padx=5)
        self.crear_toolTip(tiemq, 'Intervalo de tiempo en minutos en el que se guardaron los datos procesados')

        factorkt = ctk.CTkLabel(self, text="Factor de Seguimiento (k):").grid(row=8, column=0, sticky="w")
        factork = ctk.CTkEntry(self, textvariable=self.factork_string, width=150, validate="key", validatecommand=self.vcmd_float)
        factork.grid(row=8, column=1, sticky="w")
        factorkq = ctk.CTkButton(self, fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        factorkq.grid(row=8, column=2, sticky="w", padx=5)
        self.crear_toolTip(factorkq, 'Factor de sensibilidad para el seguimiento de hojas.\nValores menores (0.5-1.0) = seguimiento más estricto\nValores mayores (1.0-3.0) = seguimiento más permisivo')

        devicet = ctk.CTkLabel(self, text="Procesar con:").grid(row=9, column=0, sticky="w")
        self.device_menu = ctk.CTkOptionMenu(self, values=self.devices, variable=self.device_var, width=150,fg_color=("#5E7B6B", "#5E7B6B"), dropdown_fg_color=("#5E7B6B", "#5E7B6B"), dropdown_hover_color=("#397F5A", "#397F5A"), dropdown_text_color=("white", "white"))
        self.device_menu.grid(row=9, column=1, sticky="w")
        deviceq = ctk.CTkButton(self, fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        deviceq.grid(row=9, column=2, sticky="w", padx=5)
        self.crear_toolTip(deviceq, 'Selecciona el dispositivo a utilizar para realizar el procesamiento del modelo.\nCPU: Procesamiento en CPU\nCUDA: Procesamiento en GPU')

        select = ctk.CTkLabel(self, text="Carpeta de guardado").grid(row=10, column=0, sticky="w")
        boton_seleccionar_carpeta = ctk.CTkButton(self, text="Seleccionar Carpeta", command=lambda: seleccionar_carpeta(self))
        boton_seleccionar_carpeta.grid(row=10, column=1, sticky="w")
        selecq = ctk.CTkButton(self, fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        selecq.grid(row=10, column=2, sticky="w", padx=5)
        self.crear_toolTip(selecq, 'Carpeta de guardado de los datos de procesamiento')

        # Botón para seleccionar modelo
        modelo = ctk.CTkLabel(self, text="Modelo YOLO").grid(row=11, column=0, sticky="w")
        boton_seleccionar_modelo = ctk.CTkButton(self, text="Seleccionar Modelo", command=self.seleccionar_modelo)
        boton_seleccionar_modelo.grid(row=11, column=1, sticky="w")
        modeloq = ctk.CTkButton(self, fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        modeloq.grid(row=11, column=2, sticky="w", padx=5)
        self.crear_toolTip(modeloq, 'Seleccionar modelo YOLO para la detección de hojas')
        
        # Etiqueta para mostrar el modelo seleccionado
        self.modelo_label = ctk.CTkLabel(self, text="")
        self.modelo_label.grid(row=11, column=3, columnspan=2, sticky="w", padx=5)

        self.botonok = ctk.CTkButton(self, text="Confirmar", command=self.guardar)
        self.botonok.grid(row=12, column=0, sticky=W)
        self.botonok.configure(state="disabled")

        # Frame para los logos a la derecha
        self.logos_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.logos_frame.grid(row=0, column=4, rowspan=13, padx=20, pady=10, sticky="nsew")
        
        # Cargar y mostrar los logos
        self.crear_logos()

        self.pack(expand=True, fill='both')

    def crear_logos(self):
        try:
            # Logo UTN
            logo_utn = ctk.CTkImage(
                light_image=ImgPIL.open('assets/utn2.png'),
                dark_image=ImgPIL.open('assets/utn2.png'),
                size=(300, 120)
            )
            self.label_utn = ctk.CTkLabel(self.logos_frame, image=logo_utn, text="")
            self.label_utn.grid(row=0, column=0, pady=(0, 0), sticky="n")

            
            # Logo Eco Hormigas
            logo_eco = ctk.CTkImage(
                light_image=ImgPIL.open('assets/eco_hormigas.png'),
                dark_image=ImgPIL.open('assets/eco_hormigas.png'),
                size=(489, 400)
            )
            self.label_eco = ctk.CTkLabel(self.logos_frame, image=logo_eco, text="")
            self.label_eco.grid(row=1, column=0, pady=(0, 0), sticky="w")
            
        except Exception as e:
            print(f"Error al cargar los logos: {str(e)}")
            # Mostrar labels de texto si no se pueden cargar las imágenes
            self.label_utn_text = ctk.CTkLabel(self.logos_frame, text="Logo UTN", font=ctk.CTkFont(size=16))
            self.label_utn_text.grid(row=0, column=0, pady=(0, 20), sticky="w")
            
            self.label_eco_text = ctk.CTkLabel(self.logos_frame, text="Logo Eco Hormigas", font=ctk.CTkFont(size=16))
            self.label_eco_text.grid(row=1, column=0, pady=(20, 0), sticky="w")

    def cambiar_estado(self):
        self.botonok.configure(state="normal")
    
    def msgBox():
        msg.showerror('Error!', 'Error en los parametros de configuracion')
    
    def crear_toolTip(self, widget, texto):
            # Obtiene la posición del widget en la pantalla
        xTT = widget.winfo_rootx() + widget.winfo_width() + 5  # Desplazamiento hacia la derecha
        yTT = widget.winfo_rooty()
        toolTip = CTkToolTip(widget, delay=0.3, message=texto, alpha=0.3, bg_color="#000000", width=150)
        toolTip.geometry(f"+{xTT}+{yTT}")
        
    def guardar(self):      #Guardamos en el objeto configuracion los valores ingresados en las entradas
            global gv
            h, m = self.horastring.get().split(':')
            d = datetime.timedelta(hours=int(h), minutes=int(m))
            if self.fpstring.get() == 0 or self.fpsdisstring.get()== 0 or self.confstring.get()==0 or self.confstring.get()>1 or self.cantstring.get()==0 or self.tiemstring.get()==0 or self.factork_string.get()<=0:
                self.msgBox()
                return 0
            
            selected_device = self.device_var.get().split()[0]
            print(d)
            gv.configuracion = Configuracion(
                self.fechastring.get(),
                d,
                self.fpstring.get(),
                self.fpsdisstring.get(),
                self.fpsapastring.get(),
                self.confstring.get(),
                self.cantstring.get(),
                self.tiemstring.get(),
                selected_device,
                self.factork_string.get()
            )
            print(gv.configuracion)
            self.parent.habilitar_tabs()
            self.parent.siguiente("Pantalla Video")


    def validar_input(self, tipo, input):
        if tipo == 'int':
            return input.isdigit() or input == ""
        elif tipo == 'float':
            return input.replace('.', '', 1).isdigit() or input == ""    
    
    def callback(self, tipo, P):
        return self.validar_input(tipo, P)
    
    def seleccionar_modelo(self):
        global gv
        modelo_path = fd.askopenfilename(
            title='Seleccionar modelo YOLO',
            filetypes=[('Modelo PyTorch', '*.pt')],
            initialdir='src/models_data'
        )
        if modelo_path:
            gv.model_path = modelo_path
            nombre_modelo = os.path.basename(modelo_path)
            self.modelo_label.configure(text=f"✓ {nombre_modelo}", text_color="#4E8F69")

# Clase para la segunda pestaña
class Tab2(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        self.my_font = ctk.CTkFont(family="Calibri", size=14, 
                                             weight="bold") #weight bold/normal, slant=italic/roman
        self.my_font2 = ctk.CTkFont(family="Calibri", size=12, 
                                             weight="bold") #weight bold/normal, slant=italic/roman
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0,1), weight=1)

        self.FrameVideo=ctk.CTkFrame(self)
        self.FrameVideo.grid(row=0, column=0, padx=0, pady=(5, 5), sticky="nsew")

        self.FrameBtn=ctk.CTkFrame(self)
        self.FrameBtn.grid(row=1, column=0, padx=0, pady=(5, 5), sticky="nsew")

        self.FrameTxt=ctk.CTkFrame(self, width=280, height= 490, fg_color="transparent")
        self.FrameTxt.grid(row=0, column=1, padx=5, pady=(5, 5), sticky="nsew")


        # Configuración de los widgets de la pestaña 2
        self.lblVideo = ctk.CTkLabel(self.FrameVideo, text="")
        # Tamaño inicial, se ajustará automáticamente al video
        self.lblVideo.configure(width=640, height=480)
        self.lblVideo.grid(row=0, column=0,sticky="")
        self.lblVideo.bind("<Button-1>", self.on_click)

        self.progress_bar = ctk.CTkProgressBar(self.FrameVideo, orientation=HORIZONTAL, width=640, mode='determinate')
        self.progress_bar.grid(row=1, column=0, padx=(0, 10), pady=(5, 5), sticky="w")
        self.progress_bar.set(0)

        self.inicio = ctk.CTkButton(self.FrameBtn, text="Iniciar", width=96.6, command=lambda: iniciar(self))
        self.inicio.grid(row=0, column=0, padx=5, pady=(10, 10), sticky="ew")

        self.base_b = ctk.CTkButton(self.FrameBtn, text="Area de Detecion", width=96.6, command=lambda: base_blanca(self))
        self.base_b.grid(row=0, column=3, padx=5, pady=(10, 10), sticky="ew")
        self.base_b.configure(state="disabled")

        self.pausa = ctk.CTkButton(self.FrameBtn, text="Pausar", width=96.6, command=lambda: on_pause(self))
        self.pausa.grid(row=0, column=1, padx=5, pady=(10, 10), sticky="ew")
        self.pausa.configure(state="disabled")

        self.boton_seleccionES = ctk.CTkButton(self.FrameBtn, text="↑ ↓", width=96.6, command= lambda: habilitar_seleccion(self))
        self.boton_seleccionES.grid(row=0, column=2, padx=5, pady=(10, 10), sticky="ew")

        self.boton_conv = ctk.CTkButton(self.FrameBtn, text="Conversion", width=96.6, command= lambda: habilitar_conversion(self))
        self.boton_conv.grid(row=0, column=4, padx=5, pady=(10, 10), sticky="ew")

        salir = ctk.CTkButton(self.FrameBtn, hover=True, text="Salir", width=96.6, command=quit_1)
        salir.grid(row=0, column=5, padx=5, pady=(10, 10), sticky="ew")
        
        self.texto1 = ctk.CTkLabel(self.FrameTxt, text="1. Seleccione video para comenzar [Boton Iniciar]", fg_color="transparent", font=self.my_font, text_color="#abcfba")
        self.texto1.grid(row=0, column=0, padx=5, pady=(10, 10), sticky="ew")

        self.texto2 = ctk.CTkLabel(self.FrameTxt, text="2. Área de detección [Boton Área de Deteccion] \n[Click y arrastrar para dibujar un rectangulo]", fg_color="transparent", font=self.my_font, text_color="#abcfba")
        self.texto2.grid(row=1, column=0, padx=5, pady=(10, 10), sticky="ew")

        self.texto3 = ctk.CTkLabel(self.FrameTxt, text="3. Entrada y Salida\n[1er click en entrada. 2do click en salida (boca de ingreso al nido)]", fg_color="transparent", font=self.my_font, text_color="#abcfba")
        self.texto3.grid(row=2, column=0, padx=5, pady=(10, 10), sticky="ew")

        self.texto4 = ctk.CTkLabel(self.FrameTxt, text="4. Factor de conversión\n[dos clics, uno en cada esquina opuestas del cuadrado de conversion]", fg_color="transparent", font=self.my_font, text_color="#abcfba")
        self.texto4.grid(row=3, column=0, padx=5, pady=(10, 10), sticky="ew")

        self.texto_conversion = ctk.CTkLabel(self.FrameTxt, text="", fg_color="transparent", font=self.my_font, text_color="#abcfba")
        self.texto_conversion.grid(row=4, column=0, padx=5, pady=(10, 10), sticky="ew")

        self.texto5 = ctk.CTkLabel(self.FrameVideo, text="", fg_color="transparent", font=self.my_font2, text_color="#abcfba")
        self.texto5.grid(row=1, column=1, padx=(5, 0), pady=(5, 5), sticky="w")


        self.debug_var = ctk.BooleanVar(value=False)
        self.debug_checkbox = ctk.CTkCheckBox(
            self.FrameTxt, 
            text="Ventana de Detecciones", 
            variable=self.debug_var,
            command=self.toggle_debug_window,
            font=self.my_font2,
            fg_color="#4E8F69"
        )
        self.debug_checkbox.grid(row=7, column=0, padx=5, pady=(10, 10), sticky="w")

        # Menú selector para tipo de filtrado
        filtrado_label = ctk.CTkLabel(self.FrameTxt, text="Tipo de Filtrado:", font=self.my_font2)
        filtrado_label.grid(row=5, column=0, padx=5, pady=(5, 0), sticky="w")
        
        self.filtrado_var = ctk.StringVar(value="Normal")
        self.filtrado_menu = ctk.CTkOptionMenu(
            self.FrameTxt,
            values=["Normal", "Estricto"],
            variable=self.filtrado_var,
            command=self.cambiar_filtrado,
            width=150,
            font=self.my_font2,
            fg_color="#4E8F69",
            dropdown_fg_color="#4E8F69"
        )
        self.filtrado_menu.grid(row=6, column=0, padx=5, pady=(0, 10), sticky="w")

        # Crear un frame para la ventana de debug (inicialmente oculto)
        self.debug_frame = ctk.CTkFrame(self.FrameTxt)
        self.debug_frame.grid(row=8, column=0, padx=5, pady=(10, 10), sticky="nsew")
        self.debug_frame.grid_remove()  # Ocultar inicialmente

        # Label para mostrar las detecciones
        self.debug_label = ctk.CTkLabel(self.debug_frame, text="")
        self.debug_label.pack(expand=True, fill="both")

        self.pack(expand=True)

    def getlblVideo(self):
        return self.lblVideo

    def gettxt1(self):
        return self.texto1

    def gettxt2(self):
        return self.texto2

    def gettxt3(self):
        return self.texto3

    def gettxt4(self):
        return self.texto4  
    
    def gettxt5(self):
        return self.texto5

    def getTextoConversion(self):
        return self.texto_conversion

    def cambiartexto(self, widget, texto):
        widget.configure(text=texto)
    def cambiarcolor(self, widget, color):
        widget.configure(text_color=color)
    def getProgressBar(self):
        return self.progress_bar   
    
    def getPausa(self):
        return self.pausa  
    
    def getBaseBlanca(self):
        return self.base_b
    
    def getBotonSeleccionES(self):
        return self.boton_seleccionES

    def getBotonConv(self):
        return self.boton_conv
        
    def on_click(self, event):
        global gv, click_count, seleccion_entrada_habilitada, seleccion_conversion
        
        # Convertir coordenadas de pantalla a coordenadas originales
        original_x, original_y = self.display_to_original_coords(event.x, event.y)
        current_point = (original_x, original_y)
        
        # Modo de selección de área base (click y arrastre)
        if gv.bb == True:
            # Iniciar arrastre al hacer clic
            gv.point1 = current_point
            self.lblVideo.bind("<B1-Motion>", self.on_mouse_move)
            # Procesar al soltar el botón
            self.lblVideo.bind("<ButtonRelease-1>", self.on_release)
                
        # Modo de selección de entrada/salida (dos clicks)
        elif seleccion_entrada_habilitada == True:
            if click_count == 0:
                gv.point1 = current_point
                click_count += 1
            elif click_count == 1:
                gv.point2 = current_point
                click_count = 0
                self.procesar_seleccion_entrada_salida()
                
        # Modo de selección de conversión (dos clicks) 
        elif seleccion_conversion == True:
            if click_count == 0:
                gv.point1 = current_point
                click_count += 1
            elif click_count == 1:
                gv.point2 = current_point
                click_count = 0
                self.procesar_seleccion_conversion()

    def on_mouse_move(self, event):
        global gv
        
        # Convertir coordenadas de pantalla a coordenadas originales
        original_x, original_y = self.display_to_original_coords(event.x, event.y)
        gv.point2 = (original_x, original_y)
        
        # Crea una copia del frame pausado
        frame_copy = gv.img.copy()
        
        # Dibujar el rectángulo en la copia del frame usando coordenadas originales
        cv2.rectangle(frame_copy, gv.point1, gv.point2, (0, 255, 0), 2)

        # Usar tamaño original del video
        height, width = frame_copy.shape[:2]
        
        # Crear imagen para la interfaz con tamaño original
        frame_copy_pil = ImgPIL.fromarray(frame_copy)
        frame_tk = ctk.CTkImage(light_image=frame_copy_pil, dark_image=frame_copy_pil, size=(width, height))
        self.lblVideo.configure(image=frame_tk)
        self.lblVideo.image = frame_tk

    def on_release(self, event):
        global gv
        
        if gv.bb:
            # Convertir coordenadas de pantalla a coordenadas originales
            original_x, original_y = self.display_to_original_coords(event.x, event.y)
            gv.point2 = (original_x, original_y)
            
            # Limpiar los bindings
            self.lblVideo.unbind("<B1-Motion>")
            self.lblVideo.unbind("<ButtonRelease-1>")
            
            # Procesar la selección del área
            self.procesar_seleccion_area()

    def procesar_seleccion_area(self):
        # Procesar la selección del área base
        global gv, seleccion_entrada_habilitada
        
        base_blanca_aux(gv.point1, gv.point2)
        self.base_b.configure(fg_color="#4E8F69")
        

        
        self.cambiartexto(self.gettxt1(), "1. Área de deteccion ✓")
        self.cambiarcolor(self.gettxt1(), "#009929")

        self.cambiartexto(self.gettxt2(), "2. Seleccione Entrada y Salida [Boton ↑↓] ◍\n[1er click en entrada. 2do click en salida (boca de ingreso al nido)]")
        self.cambiarcolor(self.gettxt2(),  "#e5be01")

        self.cambiartexto(self.gettxt3(), "3. Seleccione el recuadro de referencia [Boton Conversion] ✘\n[dos clics, uno en cada esquina opuestas del cuadrado de referencia] ")
        self.cambiarcolor(self.gettxt3(),"#ff0800")

        self.cambiartexto(self.gettxt4(), "Configuracion incompleta - Continue los pasos ")
        self.cambiarcolor(self.gettxt4(), "#ff0800")

        self.getBotonSeleccionES().configure(state="normal", fg_color="#e5be01")

        
        # Desactivar el modo de selección de área
        gv.bb = False
        
        # Actualizar la visualización con el área seleccionada
        self._actualizar_display_con_dibujos()
        
        # Verificar si todas las configuraciones están completas
        self.verificar_configuracion_completa()

    def procesar_seleccion_entrada_salida(self):
        # Procesar la selección de entrada/salida
        global gv, seleccion_entrada_habilitada
        
        gv.entrada_coord = gv.point1
        gv.salida_coord = gv.point2
        gv.direccion = detectar_direccion_entrada_salida(gv.entrada_coord, gv.salida_coord)
        gv.entrada_salida_seleccionada = True

        self.boton_seleccionES.configure(fg_color="#4E8F69")
        self.getBotonConv().configure(state="normal", fg_color="#e5be01")

        self.cambiartexto(self.gettxt1(), "1. Área de deteccion [Boton Área de Deteccion] ✓")
        self.cambiarcolor(self.gettxt1(), "#009929")

        self.cambiartexto(self.gettxt2(), "2. Seleccione Entrada y Salida [Boton ↑↓] ✓\n[1er click en entrada. 2do click en salida (boca de ingreso al nido)]")
        self.cambiarcolor(self.gettxt2(),  "#009929")

        self.cambiartexto(self.gettxt3(), "3. Seleccione el recuadro de referencia [Boton Conversion] ◍\n[dos clics, uno en cada esquina opuestas del cuadrado de referencia] ")
        self.cambiarcolor(self.gettxt3(),"#e5be01")

        self.cambiartexto(self.gettxt4(), "Configuracion incompleta - Continue los pasos ")
        self.cambiarcolor(self.gettxt4(), "#ff0800")
        # Desactivar el modo de selección de entrada/salida
        seleccion_entrada_habilitada = False
        

        self.getBaseBlanca().configure(fg_color="#4E8F69")
        # Actualizar la visualización con los puntos
        self._actualizar_display_con_dibujos()
        
        # Verificar si todas las configuraciones están completas
        self.verificar_configuracion_completa()
    
    def procesar_seleccion_conversion(self):
        # Procesar la selección de conversión
        global gv, seleccion_conversion
        
        # Guardar los puntos de conversión
        gv.conv_p1 = gv.point1
        gv.conv_p2 = gv.point2
        
        # Calcular la distancia en píxeles entre los dos puntos seleccionados
        distance_px = math.sqrt((gv.conv_p2[0]-gv.conv_p1[0])**2 + (gv.conv_p2[1]-gv.conv_p1[1])**2)
        
        # La distancia real es siempre 10mm
        MEDIDA_REAL_MM = 10.0
        
        # Calcular factor de conversión lineal (mm/px)
        factor_lineal = MEDIDA_REAL_MM / distance_px
        
        # Calcular factor de conversión de área (mm²/px²)
        gv.cte = factor_lineal ** 2
        
        texto = "3. Conversion ✓ [%.2f mm²/px²]" % gv.cte
        
        self.cambiartexto(self.texto3, texto)
        self.boton_conv.configure(fg_color="#4E8F69")
        
        # Actualizar el nuevo widget de conversión con el valor
        texto_conversion = "Factor de Conversión: %.2f mm²/px²" % gv.cte
        self.cambiartexto(self.getTextoConversion(), texto_conversion)
        self.cambiarcolor(self.getTextoConversion(), "#4E8F69")
        
        # Indicar que la conversión ha sido seleccionada
        gv.conversion_seleccionada = True
        
        # Desactivar el modo de selección de conversión
        seleccion_conversion = False
        
        self.cambiartexto(self.gettxt1(), "1. Área de deteccion [Boton Área de Deteccion] ✓")
        self.cambiarcolor(self.gettxt1(), "#009929")

        self.cambiartexto(self.gettxt2(), "2. Seleccione Entrada y Salida [Boton ↑↓] ✓\n[1er click en entrada. 2do click en salida (boca de ingreso al nido)]")
        self.cambiarcolor(self.gettxt2(),  "#009929")

        self.cambiartexto(self.gettxt3(), "3. Seleccione el recuadro de referencia✓\n[dos clics, uno en cada esquina diagonales opuestas del cuadrado de referencia] ")
        self.cambiarcolor(self.gettxt3(),"#009929")

        self.cambiartexto(self.gettxt4(), "Configuracion Completa - Puede Presionar Play ")
        self.cambiarcolor(self.gettxt4(), "#ff0800")
        # Actualizar la visualización con el rectángulo de conversión
        self._actualizar_display_con_dibujos()
        
        # Verificar si todas las configuraciones están completas
        self.verificar_configuracion_completa()

    def verificar_configuracion_completa(self):
        # Verificar si todas las configuraciones están completas
        if gv.area_seleccionada and gv.entrada_salida_seleccionada and gv.conversion_seleccionada:
            self.pausa.configure(state="normal", fg_color="#4E8F69")
        else:
            self.pausa.configure(state="disabled", fg_color="#2B2B2B")

    def ajustar_widget_a_video(self):
        """Ajusta el tamaño del widget lblVideo al tamaño del video original"""
        if hasattr(gv, 'img') and gv.img is not None:
            height, width = gv.img.shape[:2]
            # Ajustar el widget al tamaño del video
            self.lblVideo.configure(width=width, height=height)
            # También ajustar la barra de progreso al ancho del video
            self.progress_bar.configure(width=width)
    
    def cambiar_filtrado(self, valor):
        global gv
        gv.filtrado_estricto = (valor == "Estricto")
        
    def toggle_debug_window(self):
        global gv
        gv.show_debug_window = self.debug_var.get()
        
        if gv.show_debug_window:
            # Mostrar la ventana de debug
            self.debug_frame.grid()
            # Configurar tamaño inicial
            self.debug_label.configure(width=320, height=240)
        else:
            # Ocultar la ventana de debug
            self.debug_frame.grid_remove()

    def display_to_original_coords(self, display_x, display_y):
        """Convierte coordenadas de la pantalla a coordenadas del frame original"""
        # Como ahora mostramos a tamaño original, las coordenadas son 1:1
        return display_x, display_y

    def original_to_display_coords(self, original_x, original_y):
        """Convierte coordenadas del frame original a coordenadas de pantalla"""
        # Como ahora mostramos a tamaño original, las coordenadas son 1:1
        return original_x, original_y

    def _actualizar_display_con_dibujos(self):
        """Actualiza la visualización con todos los elementos de configuración"""
        if not hasattr(gv, 'img') or gv.img is None:
            return
            
        # Crear una copia de la imagen base
        frame_copy = gv.img.copy()
        
        # Dibujar área seleccionada si existe
        if gv.area_seleccionada and hasattr(gv, 'x1'):
            cv2.rectangle(frame_copy, (gv.x1, gv.y1), (gv.x2, gv.y2), (0, 255, 0), 2)
        
        # Dibujar puntos de entrada/salida si existen
        if gv.entrada_salida_seleccionada:
            if gv.entrada_coord:
                cv2.circle(frame_copy, gv.entrada_coord, radius=5, color=(0, 255, 0), thickness=-1)
            if gv.salida_coord:
                cv2.circle(frame_copy, gv.salida_coord, radius=5, color=(255, 0, 0), thickness=-1)
        
        # Dibujar rectángulo de conversión si existe
        if gv.conversion_seleccionada and gv.conv_p1 and gv.conv_p2:
            cv2.rectangle(frame_copy, gv.conv_p1, gv.conv_p2, (0, 0, 255), 1)
        
        # Usar tamaño original del video
        height, width = frame_copy.shape[:2]
        
        # Actualizar la imagen en la interfaz con tamaño original
        im = ImgPIL.fromarray(frame_copy)
        display_img = ctk.CTkImage(light_image=im, dark_image=im, size=(width, height))
        self.lblVideo.configure(image=display_img)
        self.lblVideo.image = display_img

class Tab3(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        
        self.figure1 = None
        self.figure2 = None
        self.canvas1 = None
        self.canvas2 = None
        self.cumulative_hojas = 0  # Variable para acumular total de hojas
        
        # Configuración del frame
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Frame para controles
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        # Frame para el primer gráfico (boxplot)
        self.plot_frame1 = ctk.CTkFrame(self)
        self.plot_frame1.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # Frame para el segundo gráfico (barras)
        self.plot_frame2 = ctk.CTkFrame(self)
        self.plot_frame2.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

        # Botón para seleccionar archivo
        self.select_button = ctk.CTkButton(
            self.control_frame,
            text="Seleccionar Archivo",
            command=self.select_file
        )
        self.select_button.grid(row=0, column=0, padx=10, pady=10)

        # Label para mostrar el archivo seleccionado
        self.file_label = ctk.CTkLabel(
            self.control_frame,
            text="Ningún archivo seleccionado",
            text_color="gray"
        )
        self.file_label.grid(row=0, column=1, padx=10, pady=10)

        # Botón para crear gráfico
        self.plot_button = ctk.CTkButton(
            self.control_frame,
            text="Crear Gráficos",
            command=self.create_plots,
            state="disabled"
        )
        self.plot_button.grid(row=0, column=2, padx=10, pady=10)
        
        # Botón para guardar gráficos
        self.save_button1 = ctk.CTkButton(
            self.control_frame,
            text="Guardar Gráfico 1",
            command=lambda: self.save_plot(1),
            state="disabled"
        )
        self.save_button1.grid(row=0, column=3, padx=10, pady=10)
        
        self.save_button2 = ctk.CTkButton(
            self.control_frame,
            text="Guardar Gráfico 2",
            command=lambda: self.save_plot(2),
            state="disabled"
        )
        self.save_button2.grid(row=0, column=4, padx=10, pady=10)

        self.pack(expand=True, fill='both')

    def select_file(self):
        filename = fd.askopenfilename(
            title='Seleccionar archivo de datos',
            filetypes=[('Archivos de texto', '*.txt')]
        )
        if filename:
            self.current_file = filename
            self.file_label.configure(
                text=os.path.basename(filename),
                text_color="white"
            )
            self.plot_button.configure(state="normal")

    def create_plots(self):
        try:
            # Limpiar los gráficos anteriores si existen
            self.cleanup()
            
            # Leer el archivo CSV
            df = pd.read_csv(self.current_file)
            
            # Eliminar la primera fila si es un duplicado del encabezado
            if (df.iloc[0] == df.columns).all():
                df = df.iloc[1:]
            
            # Combinar fecha y hora para hacer una columna de tiempo
            df['Tiempo_Inicio'] = pd.to_datetime(df['Fecha'] + ' ' + df['HoraInicio'], format='%d-%m-%Y %H:%M')
            df['Tiempo_Fin'] = pd.to_datetime(df['Fecha'] + ' ' + df['HoraFin'], format='%d-%m-%Y %H:%M')
            
            # Ordenar por tiempo de inicio
            df = df.sort_values('Tiempo_Inicio')
            
            # Crear etiquetas para el eje X en formato "HH:MM-HH:MM"
            labels = [f"{inicio.strftime('%H:%M')}-{fin.strftime('%H:%M')}" for inicio, fin in zip(df['Tiempo_Inicio'], df['Tiempo_Fin'])]
            
            # GRÁFICO 1: Boxplot (Gráfico de velas)
            self.create_boxplot(df, labels)
            
            # GRÁFICO 2: Cantidad de hojas y área total acumulada
            self.create_hojas_plot(df, labels)
            
            # Habilitar botones de guardado
            self.save_button1.configure(state="normal")
            self.save_button2.configure(state="normal")
            
        except Exception as e:
            msg.showerror('Error', f'Error al crear los gráficos: {str(e)}')
            print(f"Error detallado: {str(e)}")
    
    def create_boxplot(self, df, labels):
        # Crear figura para el boxplot
        self.figure1 = plt.figure(figsize=(12, 5))
        ax = self.figure1.add_subplot(111)
        
        # Datos para el boxplot
        data = []
        for _, row in df.iterrows():
            stats = {
                'whislo': row['Minimo'],      # Mínimo
                'q1': row['Percentil25'],     # Q1 (Percentil 25)
                'med': row['Mediana'],        # Mediana
                'q3': row['Percentil75'],     # Q3 (Percentil 75)
                'whishi': row['Maximo'],      # Máximo
                'fliers': []                  # Sin outliers
            }
            data.append(stats)
        
        # Crear el boxplot
        bplot = ax.bxp(data, patch_artist=True)
        
        # Personalizar el boxplot
        for patch in bplot['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        for median in bplot['medians']:
            median.set_color('darkblue')
            median.set_linewidth(1.5)
        
        # Configurar el gráfico
        ax.set_title('Distribución del área de hojas por intervalo', fontsize=14)
        ax.set_xlabel('Intervalos de tiempo', fontsize=12)
        ax.set_ylabel('Área (mm²)', fontsize=12)
        
        # Mostrar todos los intervalos en el eje X
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        
        # Asegurar que se vean todos los ticks del eje X
        plt.subplots_adjust(bottom=0.2)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Ajustar la figura
        plt.tight_layout()
        
        # Crear el canvas para mostrar el gráfico
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas1 = FigureCanvasTkAgg(self.figure1, master=self.plot_frame1)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(fill='both', expand=True)
    
    def create_hojas_plot(self, df, labels):
        # Crear figura para el gráfico de hojas
        self.figure2 = plt.figure(figsize=(12, 5))
        ax1 = self.figure2.add_subplot(111)
        
        # Crear el gráfico de barras para cantidad de hojas
        x = range(1, len(df) + 1)
        bars = ax1.bar(x, df['CantHojas'], color='lightgreen', alpha=0.7, label='Cantidad de Hojas')
        
        # Asegurarse de que el eje Y tenga el rango adecuado para los valores de CantHojas
        max_hojas = df['CantHojas'].max()
        ax1.set_ylim(0, max_hojas * 1.1)  # Dar un 10% extra de espacio
        
        # Configurar el primer eje Y
        ax1.set_xlabel('Intervalos de tiempo', fontsize=12)
        ax1.set_ylabel('Cantidad de Hojas', fontsize=12, color='green')
        ax1.tick_params(axis='y', labelcolor='green')
        
        # Crear un segundo eje Y para el área total por intervalo
        ax2 = ax1.twinx()
        
        # Usar directamente los valores de AreaTotal (sin acumular)
        ax2.plot(x, df['AreaTotal'], 'r-', marker='o', linewidth=2, label='Área Total')
        
        # Asegurarse de que el eje Y2 tenga el rango adecuado para los valores de área
        max_area = df['AreaTotal'].max()
        ax2.set_ylim(0, max_area * 1.1)  # Dar un 10% extra de espacio
        
        ax2.set_ylabel('Área Total (mm²)', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Asegurar que se muestren todos los valores en el eje X
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        
        # Asegurar que se vean todos los ticks del eje X
        plt.subplots_adjust(bottom=0.2)
        
        # Añadir título y leyenda
        ax1.set_title('Cantidad de hojas y área total por intervalo', fontsize=14)
        
        # Crear leyenda combinada
        lines, labels_leg = ax1.get_legend_handles_labels()
        lines2, labels_leg2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels_leg + labels_leg2, loc='upper left')
        
        # Ajustar la figura
        plt.tight_layout()
        
        # Crear el canvas para mostrar el gráfico
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas2 = FigureCanvasTkAgg(self.figure2, master=self.plot_frame2)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(fill='both', expand=True)
    
    def save_plot(self, plot_num):
        try:
            # Definir opciones para guardar
            file_types = [('PNG', '*.png'), ('JPEG', '*.jpg'), ('PDF', '*.pdf')]
            default_name = f"grafico_{plot_num}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Solicitar ubicación de guardado
            filename = fd.asksaveasfilename(
                title=f'Guardar Gráfico {plot_num}',
                defaultextension=".png",
                filetypes=file_types,
                initialfile=default_name
            )
            
            if filename:
                if plot_num == 1 and self.figure1:
                    self.figure1.savefig(filename, dpi=300, bbox_inches='tight')
                    msg.showinfo('Éxito', f'Gráfico 1 guardado como {os.path.basename(filename)}')
                elif plot_num == 2 and self.figure2:
                    self.figure2.savefig(filename, dpi=300, bbox_inches='tight')
                    msg.showinfo('Éxito', f'Gráfico 2 guardado como {os.path.basename(filename)}')
        
        except Exception as e:
            msg.showerror('Error', f'Error al guardar el gráfico: {str(e)}')
    
    def cleanup(self):
        """Limpia los recursos de los gráficos"""
        if self.canvas1:
            self.canvas1.get_tk_widget().destroy()
        if self.figure1:
            plt.close(self.figure1)
            
        if self.canvas2:
            self.canvas2.get_tk_widget().destroy()
        if self.figure2:
            plt.close(self.figure2)
            
        # Limpiar los frames de gráficos
        for widget in self.plot_frame1.winfo_children():
            widget.destroy()
        for widget in self.plot_frame2.winfo_children():
            widget.destroy()

def quit_1():   #Funcion que cierra la ventana principal
     
    app.destroy()
    app.quit()


# Bucle de ejecucion de la ventana.
app = App()
app.mainloop()

# Release the video capture object and close the display window
gv.cap.release()
cv2.destroyAllWindows()
