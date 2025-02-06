
import cv2
import numpy as np
import torch
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

from src.modules.hoja import Hoja, posicion, comparar, Aparicion, xycentro, xypredic
from src.modules.ToolTip import *
from src.modules.KalmanFilter import KalmanFilter

#Defino la variable global
class VGlobals:
    def __init__(self):
        self.PASSES=None
        
gv=VGlobals()


class Configuracion:    #Clase de configuracion
    def __init__(self, fecha, hora, fps, fpsdist, fpsapa, conf, cantapa, tiempo, device):
        self.fecha = fecha
        self.hora = hora
        self.fps = fps
        self.fpsdist = fpsdist
        self.fpsapa = fpsapa
        self.conf = conf
        self.cantapa = cantapa
        self.tiempo = tiempo
        self.device = device  

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
    
    def __str__(self):
        return f"Config: fecha:{self.fecha}, hora:{self.hora}, FPS:{self.fps},FPSDist:{self.fpsdist}, FPSAp:{self.fpsapa}, Confianza:{self.conf}, Cantidad de apariciones: {self.cantapa}, Tiempo de guardado: {self.tiempo}, Dispositivo: {self.device}"

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
    gv.filename = gv.filenames[0]
    gv.hojas.clear()
    device = gv.configuracion.device
    
    gv.archi1 = open(os.path.join(gv.carpeta_seleccionada, "datos-" + str(gv.configuracion.getfecha()) + ".txt"), "w+")
    gv.archi2 = open(os.path.join(gv.carpeta_seleccionada, "intervalo-" + str(gv.configuracion.getfecha()) + ".txt"), "w+")
    gv.archi2.write("Cant Hojas|Mediana|Percentil 25|Percentil 75|Minimo|Maximo|TSE|CI|Hora|\n")
    
    gv.ID=-1
    # Elegimos la camara
    gv.model = YOLO("src/models_data/200_Grayscale.pt")
    gv.model.to(device)

    gv.cap = cv2.VideoCapture(gv.filename)
    print(gv.configuracion)
    visualizar(UI2)
    on_pause(UI2)
    pausa=UI2.getPausa()
    base_b=UI2.getBaseBlanca()
    UI2.cambiartexto(UI2.gettxt2(), "Seleccionar el área de detección")
    
    UI2.getPausa().configure(state= "disabled")
    UI2.getBaseBlanca().configure(state="normal", fg_color="#f56767")
    #captura.config(state="normal")
    
    

def on_pause(UI2):
    global gv
    gv.paused = not gv.paused
    pausa=UI2.getPausa()
    if gv.paused == True:
        pausa.configure(text="Pausa")
    else:
        pausa.configure(text="Play")
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
    global gv, primera
    gv.bb=True
    primera = True
    gv.paused=True
    #text4.config(text="Esperando area de detección")
    visualizar(UI2)

def base_blanca_aux(point1, point2): # Aca nada mas cargamos las variables y cambia el texto
    global bb, gv
    gv.y1=point1[1]
    gv.x1=point1[0]
    gv.y2=point2[1]
    gv.x2=point2[0]
    gv.yinicio=gv.y2 - 20
    gv.yfinal= gv.y1 + 20
    
def habilitar_seleccion(UI2):
    global gv, seleccion_entrada_habilitada, primera
    primera = True
    pausa=UI2.getPausa()
    pausa.configure(text="Play")
    seleccion_entrada_habilitada = not seleccion_entrada_habilitada
    gv.paused=True
    visualizar(UI2)

def habilitar_conversion(UI2):
    global seleccion_conversion, primera
    primera = True
    pausa=UI2.getPausa()
    pausa.configure(text="Play")
    seleccion_conversion = not seleccion_conversion
    gv.paused=True
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
                    if (primer_aparicion.gety() - ultima_aparicion.gety()) > 30:
                        hoja.valid_id = gv.valid_ID
                        gv.hojas_final.append(hoja)
                        gv.hojas_final = filtrar_duplicados(gv.hojas_final)
                elif bandera_superada==False:
                    hoja.valid_id = gv.valid_ID
                    gv.hojas_final_sale.append(hoja)
            # Eliminar la hoja del array hojas
            del hojas[i]
    return hojas
    
def finalizar():
    global gv
    gv.cap.release()
    cv2.destroyAllWindows()
    print("FIN")
  
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
    global gv
    gv.estadisticas = []
    
    if bandera == 0:
        for item in hojas_final:
            for aparicion in item.apariciones:
                gv.archi1.write(str(item.valid_id)+"|"+str(aparicion.getx()) +"|"+ 
                               str(aparicion.gety()) +"|"+ str(aparicion.getxp()) +"|"+ 
                               str(aparicion.getyp()) +"|"+str(aparicion.getarea())+ "|"+ 
                               str(aparicion.getframe())+"\n")
    
    if bandera == 1:
        gv.archi2.seek(0, 2)
        
        # Obtener estadísticas de las hojas en el intervalo de tiempo
        gv.estadisticas = [item.getarea() for item in hojas_final 
                          if (gv.garch - gv.configuracion.gettiempo()) < 
                          (item.apariciones[0].getframe() / (30 * 60)) < gv.garch] 

        if len(gv.estadisticas) > 0:
            area_mediana = np.mean([item['mediana'] for item in gv.estadisticas])*gv.cte
            area_percentil25 = np.mean([item['percentil25'] for item in gv.estadisticas])*gv.cte
            area_percentil75 = np.mean([item['percentil75'] for item in gv.estadisticas])*gv.cte
            area_maxima = np.max([item['maximo'] for item in gv.estadisticas])*gv.cte
            area_minima = np.min([item['minimo'] for item in gv.estadisticas])*gv.cte
            
            # Calcular métricas de seguimiento
            metricas = evaluar_seguimiento(hojas_final)
            tse = metricas["Tasa_Seguimiento_Exitoso"]
            tct = metricas["Tasa_de_Trayectorias"]

            minutos_transcurridos = int(gv.garch)
            hora_inicial = gv.configuracion.gethora()
            hora_actual = hora_inicial + datetime.timedelta(minutes=minutos_transcurridos)
            hora_str = f"{hora_actual.days * 24 + hora_actual.seconds // 3600:02d}:{(hora_actual.seconds // 60) % 60:02d}"


            # Escribir todos los valores incluyendo máximo y mínimo
            gv.archi2.write(f"{len(gv.estadisticas)}|{area_mediana:.2f}|{area_percentil25:.2f}|"
               f"{area_percentil75:.2f}|{area_minima:.2f}|{area_maxima:.2f}|"
               f"{tse:.2f}|{tct:.2f}|" 
               f"{hora_str}\n")
            
            if not hasattr(gv, 'total_hojas'):
                gv.total_hojas = 0
            gv.total_hojas += len(hojas_final)
            
            # Limpiar la lista de hojas
            gv.hojas_final.clear()


    # archi1.write("Saliente: \n")
    # for item in hojas_final_sale:
    #     for aparicion in item.apariciones:
    #         archi1.write(str(item.id)+ "|"+str(aparicion.getx()) +"|"+ str(aparicion.gety()) +"|"+ str(aparicion.getarea())+"|"+str(aparicion.getframe())+"\n")

    

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

            # Procesar frame
            frame = img[gv.y1:gv.y2, gv.x1:gv.x2]
            frame = rotar_imagen(frame, gv.direccion)

            # Procesar detecciones YOLO solo cuando sea necesario
            if gv.frameactual - gv.frameaux >= gv.configuracion.getfpsdist():
                results = gv.model.predict(frame, conf=gv.configuracion.getconf())
                gv.annotated_frame = results[0].plot()
                detector(results, gv.frameactual)
                gv.frameaux = gv.frameactual

            # Actualizar tiempo y textos
            sec = gv.frameactual / gv.configuracion.getfps()
            s = datetime.timedelta(seconds=int(sec))
            hora = gv.configuracion.gethora() + s
            
            # Actualizar UI
            UI2.cambiartexto(UI2.gettxt5(), str(hora))
            UI2.cambiartexto(UI2.gettxt4(), f"Hojas entrantes: {gv.total_hojas + len(gv.hojas_final)}")

            # Manejo de guardado automático
            gv.garch = sec / 60
            # Asegurarse de que gettiempo() retorna un número
            tiempo_guardado = float(gv.configuracion.gettiempo())
            if tiempo_guardado > 0 and (gv.garch % gv.configuracion.gettiempo()) == 0:  # Usar una pequeña tolerancia
                escribirarchivo(gv.hojas_final, gv.hojas_final_sale, 0)
                escribirarchivo(gv.hojas_final, gv.hojas_final_sale, 1)

            # Procesar hojas
            gv.hojas = eliminar_hojas(gv.hojas, gv.frameactual)

            # Dibujar elementos visuales
            cv2.rectangle(img, (gv.x1, gv.y1), (gv.x2, gv.y2), (0, 255, 0), 2)
            cv2.line(img, (gv.x1, gv.yfinal), (gv.x2, gv.yfinal), (255, 255, 255), 1)
            cv2.line(img, (gv.x1, gv.yinicio), (gv.x2, gv.yinicio), (255, 255, 255), 1)
            cv2.circle(img, gv.entrada_coord, radius=5, color=(0, 255, 0), thickness=-1)
            cv2.circle(img, gv.salida_coord, radius=5, color=(0, 0, 255), thickness=-1)

            # Convertir y preparar imagen para mostrar
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gv.img = img_rgb.copy()
            img_resized = imutils.resize(img_rgb, width=640)
            im = ImgPIL.fromarray(img_resized)
            display_img = ctk.CTkImage(light_image=im, dark_image=im, size=(640, 480))

            # Actualizar video en UI
            lblVideo = UI2.getlblVideo()
            lblVideo.configure(image=display_img)
            lblVideo.image = display_img

            # Actualizar barra de progreso
            current_frame = gv.cap.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = gv.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            progress = current_frame / total_frames
            UI2.getProgressBar().set(progress)

            # Programar siguiente frame
            lblVideo.after(10, lambda: visualizar(UI2))

        except Exception as e:
            print(f"Error en visualizar: {e}")
            if gv.cap is not None:
                gv.cap.release()
            cv2.destroyAllWindows()

def handle_end_of_video(UI2, gv):
    """
    Maneja el final del video actual
    """
    gv.cap.release()
    if gv.filenames.index(gv.filename) == len(gv.filenames)-1:
        gv.archi1.close()
        gv.archi2.close()
        cv2.destroyAllWindows()
    else:
        gv.filename = gv.filenames[gv.filenames.index(gv.filename)+1]
        gv.cap = cv2.VideoCapture(gv.filename)
        visualizar(UI2)


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
        self.geometry("1024x640")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("assets/marsh.json")
        
        # Configuración de las pestañas
        self.pestanias = MyTabView(master=self)
        self.pestanias.pack(expand=True, fill='both')

        # Agregar protocolo de cierre
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

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
        #self.configure(state="disabled")
        self.add("Init")
        self.add("Pantalla Video")
        self.add("Análisis")

        # add widgets on tabs
        self.tab("Init").configure(border_width=0)  # Opcional, para estandarizar el estilo
        Tab1(self.tab("Init"), parent=self)

        self.tab("Pantalla Video").configure(border_width=0)  # Opcional
        Tab2(self.tab("Pantalla Video"))

        self.tab("Análisis").configure(border_width=0)
        Tab3(self.tab("Análisis"))  # Nueva clase Tab3
        

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

        devicet = ctk.CTkLabel(self, text="Dispositivo:").grid(row=8, column=0, sticky="w")
        self.devices = ['cpu']  # Siempre incluir CPU
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.devices.append(f"cuda:{i} ({torch.cuda.get_device_name(i)})")
        
        # Variable para almacenar el dispositivo seleccionado
        self.device_var = ctk.StringVar(value=self.devices[0])
        
        # Menú desplegable para dispositivos
        self.device_menu = ctk.CTkOptionMenu(self, values=self.devices, variable=self.device_var, width=150,fg_color=("#5E7B6B", "#5E7B6B"), dropdown_fg_color=("#5E7B6B", "#5E7B6B"), dropdown_hover_color=("#397F5A", "#397F5A"), dropdown_text_color=("white", "white"))
        self.device_menu.grid(row=8, column=1, sticky="w")

        deviceq = ctk.CTkButton(self, fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        deviceq.grid(row=8, column=2, sticky="w", padx=5)
        self.crear_toolTip(deviceq, 'Selecciona el dispositivo de procesamiento para el modelo.\nCPU: Procesamiento en CPU\nCUDA: Procesamiento en GPU')

        # Funciones de validación
        self.vcmd_int = (self.register(lambda P: self.callback('int', P)), '%P')
        self.vcmd_float = (self.register(lambda P: self.callback('float', P)), '%P')
        
        
        #Widgets de fecha
        fechat = ctk.CTkLabel(self, text="Fecha:")
        fechat.grid(row=0, column=0, sticky="w")
        fecha = ctk.CTkEntry(self, textvariable=self.fechastring, width=150)
        fecha.grid(row=0, column=1, sticky="w")
        fechaq = ctk.CTkButton(self, fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        fechaq.grid(row=0, column=2, sticky="w", padx=5)

        self.crear_toolTip(fechaq, 'Fecha del video en formato DD-MM-YYYY')
        
        #Widget de hora
        horat = ctk.CTkLabel(self, text="Hora:").grid(row=1, column=0, sticky=W)
        hora = ctk.CTkEntry(self, textvariable = self.horastring, width=150).grid(row=1, column=1, sticky=W)
        horaq = ctk.CTkButton(self,fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        horaq.grid(row=1, column=2, sticky=W, padx=5)
        self.crear_toolTip(horaq, 'Hora de inicion del video en formato HH:MM')

        fpst = ctk.CTkLabel(self, text="FPS:").grid(row=2, column=0, sticky="w")
        FPS = ctk.CTkEntry(self, textvariable=self.fpstring, width=150, validatecommand=self.vcmd_int)
        FPS.grid(row=2, column=1, sticky="w")
        fpsq = ctk.CTkButton(self, fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        fpsq.grid(row=2, column=2, sticky="w", padx=5)
        self.crear_toolTip(fpsq, 'FPS del vídeo')
        
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

        tiemt = ctk.CTkLabel(self, text="Tiempo de guardado:").grid(row=7, column=0, sticky="w")
        tiem = ctk.CTkEntry(self, textvariable=self.tiemstring, width=150, validate="key", validatecommand=self.vcmd_float)
        tiem.grid(row=7, column=1, sticky="w")
        tiemq = ctk.CTkButton(self, fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        tiemq.grid(row=7, column=2, sticky="w", padx=5)
        self.crear_toolTip(tiemq, 'Intervalo de tiempo en minutos en el que se guardaron los datos procesados')

        select = ctk.CTkLabel(self, text="Carpeta de guardado").grid(row=9, column=0, sticky="w")
        boton_seleccionar_carpeta = ctk.CTkButton(self, text="Seleccionar Carpeta", command=lambda: seleccionar_carpeta(self))
        boton_seleccionar_carpeta.grid(row=9, column=1, sticky="w")
        selecq = ctk.CTkButton(self, fg_color="transparent", image=imagenQ, text="", height=16, width=16)
        selecq.grid(row=9, column=2, sticky="w", padx=5)
        self.crear_toolTip(selecq, 'Carpeta de guardado de los datos de procesamiento')

        self.botonok = ctk.CTkButton(self, text="Confirmar", command=self.guardar)
        self.botonok.grid(row=10, column=0, sticky=W)
        self.botonok.configure(state="disabled")

        self.pack(expand=True, fill='both')

    def cambiar_estado(self):
        self.botonok.configure(state="normal")
    
    def msgBox():
        msg.showerror('Error!', 'Error en los parametros de configuracion')
    
    def crear_toolTip(self, widget, texto):
        toolTip = CTkToolTip(widget, delay=0.3, message=texto, alpha=0.3, bg_color="#000000", width=150)
        
    def guardar(self):      #Guardamos en el objeto configuracion los valores ingresados en las entradas
            global gv
            h, m = self.horastring.get().split(':')
            d = datetime.timedelta(hours=int(h), minutes=int(m))
            if self.fpstring.get() == 0 or self.fpsdisstring.get()== 0 or self.confstring.get()==0 or self.confstring.get()>1 or self.cantstring.get()==0 or self.tiemstring.get()==0:
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
                selected_device
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
    

# Clase para la segunda pestaña
class Tab2(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        self.my_font = ctk.CTkFont(family="Calibri", size=18, 
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
        self.lblVideo.configure(width=640, height=480)
        self.lblVideo.grid(row=0, column=0,sticky="")
        self.lblVideo.bind("<Button-1>", self.on_click)

        self.progress_bar = ctk.CTkProgressBar(self.FrameVideo, orientation=HORIZONTAL, width=640, mode='determinate')
        self.progress_bar.grid(row=1, column=0, padx=0, pady=(5, 5), sticky="")
        self.progress_bar.set(0)

        self.inicio = ctk.CTkButton(self.FrameBtn, text="Iniciar", width=96.6, command=lambda: iniciar(self))
        self.inicio.grid(row=0, column=0, padx=5, pady=(10, 10), sticky="ew")

        self.base_b = ctk.CTkButton(self.FrameBtn, text="Area de Detec.", width=96.6, command=lambda: base_blanca(self))
        self.base_b.grid(row=0, column=3, padx=5, pady=(10, 10), sticky="ew")
        self.base_b.configure(state="disabled")

        self.pausa = ctk.CTkButton(self.FrameBtn, text="Pausar", width=96.6, command=lambda: on_pause(self))
        self.pausa.grid(row=0, column=1, padx=5, pady=(10, 10), sticky="ew")
        self.pausa.configure(state="disabled")

        self.boton_seleccion = ctk.CTkButton(self.FrameBtn, text="↑ ↓", width=96.6, command= lambda: habilitar_seleccion(self))
        self.boton_seleccion.grid(row=0, column=2, padx=5, pady=(10, 10), sticky="ew")
        self.boton_seleccion.configure(state="disabled")

        self.boton_conv = ctk.CTkButton(self.FrameBtn, text="Conversion", width=96.6, command= lambda: habilitar_conversion(self))
        self.boton_conv.grid(row=0, column=4, padx=5, pady=(10, 10), sticky="ew")


        salir = ctk.CTkButton(self.FrameBtn, hover=True, text="Salir", width=96.6, command=quit_1)
        salir.grid(row=0, column=5, padx=5, pady=(10, 10), sticky="ew")
        

        self.texto1 = ctk.CTkLabel(self.FrameTxt, text="", fg_color="transparent", font=self.my_font, text_color="#abcfba")
        self.texto1.grid(row=0, column=0, padx=5, pady=(10, 10), sticky="ew")

        self.texto2 = ctk.CTkLabel(self.FrameTxt, text="", fg_color="transparent", font=self.my_font, text_color="#abcfba")
        self.texto2.grid(row=1, column=0, padx=5, pady=(10, 10), sticky="ew")

        self.texto3 = ctk.CTkLabel(self.FrameTxt, text="", fg_color="transparent", font=self.my_font, text_color="#abcfba")
        self.texto3.grid(row=2, column=0, padx=5, pady=(10, 10), sticky="ew")

        self.texto4 = ctk.CTkLabel(self.FrameTxt, text="", fg_color="transparent", font=self.my_font, text_color="#abcfba")
        self.texto4.grid(row=3, column=0, padx=5, pady=(10, 10), sticky="ew")

        self.texto5 = ctk.CTkLabel(self.FrameVideo, text="", fg_color="transparent", font=self.my_font2, text_color="#abcfba")
        self.texto5.grid(row=1, column=1, padx=5, pady=(10, 10), sticky="ew")

        
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

    def cambiartexto(self, widget, texto):
        widget.configure(text=texto)
    
    def getProgressBar(self):
        return self.progress_bar   
    
    def getPausa(self):
        return self.pausa  
    
    def getBaseBlanca(self):
        return self.base_b
    
    def getBotonSeleccion(self):
        return self.boton_seleccion

    def on_click(self, event):
            global gv
            global click_count, point1, point2
            if click_count==0 :
                gv.point1 = (event.x, event.y)
                click_count += 1
                self.lblVideo.bind("<B1-Motion>", self.on_mouse_move)
                if gv.bb == True:
                    self.lblVideo.bind("<ButtonRelease-1>", self.on_click)
            elif click_count==1:
                self.lblVideo.unbind("<B1-Motion>")
                gv.point2 = (event.x, event.y)
                click_count += 1
                self.interacciones_click()

                
    def interacciones_click(self):  
        global gv, primera, seleccion_entrada_habilitada, click_count, seleccion_conversion     
        if gv.bb==True:
            base_blanca_aux(gv.point1, gv.point2)
            on_pause(self)
            self.base_b.configure(fg_color="#4E8F69")
            self.pausa.configure(state= "normal")
            on_pause(self)
            primera = False
            gv.bb = False
            self.cambiartexto(self.texto1, "Area seleccionada")
            self.cambiartexto(self.texto2, "Seleccione la entrada y salida del hormiguero")
            self.boton_seleccion.configure(state="normal", fg_color="#f56767")
            click_count = 0
            self.lblVideo.unbind("<ButtonRelease-1>")
        elif seleccion_entrada_habilitada==True:
            gv.entrada_coord = gv.point1
            gv.salida_coord = gv.point2
            gv.direccion=detectar_direccion_entrada_salida(gv.entrada_coord, gv.salida_coord)
            self.boton_seleccion.configure(fg_color="#4E8F69")
            on_pause(self)
            self.cambiartexto(self.texto2, "Entrada y salida seleccionada")
            self.cambiartexto(self.texto3, "Seleccione area de equivalencia")
            self.boton_conv.configure(state="normal", fg_color="#f56767")
            on_pause(self)
            primera = False
            seleccion_entrada_habilitada=False
            click_count = 0
        elif seleccion_conversion==True:
            distance = math.sqrt((gv.point2[0]-gv.point1[0])**2 + (gv.point2[1]-gv.point1[1])**2)
            # Display distance on GUI
            gv.cte=(10**2)/(distance**2)
            texto= "Conv: "+ "%.2f" %gv.cte+"[mm^2/px^2]"
            self.cambiartexto(self.texto3, texto)
            self.boton_conv.configure(fg_color="#4E8F69")
            on_pause(self)
            seleccion_conversion=False
            click_count = 0

    def on_mouse_move(self, event):
        global gv
        
        # Actualiza la posicion del segundo punto (esquina opuesta)
        gv.point2 = (event.x, event.y)
        
        # Crea una copia del frame pausado
        frame_copy = gv.img.copy()
        
        # Dibujar el rectangulo en la copia del frame
        cv2.rectangle(frame_copy, gv.point1, gv.point2, (0, 255, 0), 2)

        frame_copy=ImgPIL.fromarray(frame_copy)
        # Actualizar la imagen en la interfaz grafica
        frame_tk = ctk.CTkImage(light_image=frame_copy, dark_image=frame_copy, size=(640,480))
        self.lblVideo.configure(image=frame_tk)
        self.lblVideo.image = frame_tk  

class Tab3(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        
        self.figure = None
        self.canvas = None

        # Configuración del frame
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Frame para controles
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        # Frame para el gráfico (usando matplotlib)
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

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
            text="Crear Gráfico",
            command=self.create_plot,
            state="disabled"
        )
        self.plot_button.grid(row=0, column=2, padx=10, pady=10)

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

    def create_plot(self):
        try:
            # Limpiar el gráfico anterior si existe
            if self.canvas:
                self.canvas.get_tk_widget().destroy()
            if self.figure:
                plt.close(self.figure)

            # Limpiar el frame del gráfico
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            # Leer el archivo con los nombres de columnas exactos
            df = pd.read_csv(self.current_file, sep='|')
            
            # Ordenar por hora
            df = df.sort_values('Hora')
            
            data = []
            labels = []
            cant_hojas = []
            
            for _, row in df.iterrows():
                stats = [
                    float(row['Minimo']), 
                    float(row['Percentil 25']), 
                    float(row['Mediana']), 
                    float(row['Percentil 75']), 
                    float(row['Maximo'])
                ]
                data.append(stats)
                labels.append(f'{row["Hora"]}')  # Usar la hora directamente
                cant_hojas.append(int(row['Cant Hojas']))

            # Crear figura de matplotlib
            self.figure = plt.figure(figsize=(10, 6))
            ax1 = self.figure.add_subplot(111)

            # Crear el boxplot
            bplot = ax1.bxp(
                [{
                    'whislo': stats[0],    # Mínimo
                    'q1': stats[1],        # Q1
                    'med': stats[2],       # Mediana
                    'q3': stats[3],        # Q3
                    'whishi': stats[4],    # Máximo
                    'fliers': []           # Sin outliers
                } for stats in data],
                patch_artist=True
            )

            # Personalizar el gráfico
            plt.title('Distribución por Hora', pad=20)
            plt.xlabel('Hora')
            plt.ylabel('Área (mm²)')
            
            ax2 = ax1.twinx()
            x_points = range(1, len(labels) + 1)
            ax2.plot(x_points, cant_hojas, 'r-', marker='o', linewidth=2, label='Cantidad de Hojas')
            ax2.set_ylabel('Cantidad de Hojas', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.grid(False)  # Desactivar grid para el segundo eje

            # Configurar las etiquetas del eje X
            ax1.set_xticks(range(1, len(labels) + 1))
            ax1.set_xticklabels(labels, rotation=45)

            # Colorear las cajas
            for box in bplot['boxes']:
                box.set_facecolor('lightblue')
                box.set_alpha(0.7)
            
            # Configurar las etiquetas del eje X
            plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
            
            # Agregar grid
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Crear el widget de matplotlib
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

        except Exception as e:
            msg.showerror('Error', f'Error al crear el gráfico: {str(e)}')
            # Para debug
            print(f"Error detallado: {str(e)}")
            import traceback
            traceback.print_exc()

    def cleanup(self):
        """Limpia los recursos del gráfico"""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.figure:
            plt.close(self.figure)

def quit_1():   #Funcion que cierra la ventana principal
#     #finalizar()
     
     app.destroy()
     app.quit()
     #exit()
    


# Bucle de ejecucion de la ventana.
app = App()
app.mainloop()

# Release the video capture object and close the display window
gv.cap.release()
cv2.destroyAllWindows()
