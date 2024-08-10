
import cv2
import numpy as np
import torch
import imutils
import math
import os
import random
import datetime
import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog as fd
from tkinter import messagebox as msg
from tkinter.messagebox import showinfo
from tkinter.ttk import Notebook, Frame
from time import sleep
from ultralytics import YOLO
from PIL import Image as ImgPIL
from PIL import ImageTk

from src.modules.hoja import Hoja, posicion, comparar, Aparicion, xycentro, xypredic
from src.modules.ToolTip import *
from src.modules.KalmanFilter import KalmanFilter

#Defino la variable global
class VGlobals:
    def __init__(self):
        self.PASSES=None
        
gv=VGlobals()


class Configuracion:    #Clase de configuracion
    def __init__(self, fecha, hora, fps, fpsdist, fpsapa, conf, cantapa, tiempo):
        self.fecha = fecha
        self.hora = hora
        self.fps = fps
        self.fpsdist = fpsdist
        self.fpsapa = fpsapa
        self.conf = conf
        self.cantapa = cantapa
        self.tiempo = tiempo

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
    
    def __str__(self):
        return f"Config: fecha:{self.fecha}, hora:{self.hora}, FPS:{self.fps},FPSDist:{self.fpsdist}, FPSAp:{self.fpsapa}, Confianza:{self.conf}, Cantidad de apariciones: {self.cantapa}, Tiempo de guardado: {self.tiempo}"

    def __json__(self):
        return '["config": {"fecha":{self.fecha}, "hora":{self.hora}, "FPS":{self.fps},"FPSDist":{self.fpsdist}, "FPSAp":{self.fpsapa}, "Confianza":{self.conf}, "CantidadApariciones": {self.cantapa}, "DeltaTiempo": {self.tiempo}},"data":]'


def calculararea(hojas_final):
    area=0
    for hoja in hojas_final:
        area += hoja.getarea()
    return area


def seleccionar_carpeta():
    carpeta = fd.askdirectory()
    if carpeta:
        gv.carpeta_seleccionada = carpeta
        botonok.config(state="normal")

def iniciar():
    global gv
    
    gv.filenames = fd.askopenfilename(multiple=True, title='Seleccione los videos')
    gv.filename = gv.filenames[0]
    gv.hojas.clear()
    
    
    gv.archi1 = open(os.path.join(gv.carpeta_seleccionada, "salidas/datos-" + str(gv.configuracion.getfecha()) + ".txt"), "w+")
    gv.archi2 = open(os.path.join(gv.carpeta_seleccionada, "salidas/intervalo-" + str(gv.configuracion.getfecha()) + ".txt"), "w+")
    gv.archi2.write("Cant Hojas|Mediana|Percentil 25| Percentil 75| Hora\n")
    
    gv.ID=-1
    # Elegimos la camara
    gv.model = YOLO("src/models_data/100_372.pt")

    gv.cap = cv2.VideoCapture(gv.filename)
    print(gv.configuracion)
    
    visualizar()
    on_pause()
    pausa.config(state= DISABLED)
    base_b.config(state="normal")
    captura.config(state="normal")
    
    

def on_pause():
    global gv
    gv.paused = not gv.paused
    if gv.paused == True:
        pausa.config(image = imagenBI)
    else:
        pausa.config(image = imagenBF)
    visualizar()

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
                comparar(dx, dy, xmed, ymed, area, frameactual,gv,kf)

def capturar():
    global gv
    if gv.cap is not None:
        success, img = gv.cap.read()
        if success:
            # Crear la carpeta screenshots si no existe
            if not os.path.exists("salidas/screenshots"):
                os.makedirs("salidas/screenshots")
            # Obtener el listado de archivos en la carpeta screenshots
            files = os.listdir("salidas/screenshots")
            # Si no existen capturas, crear la primera
            if not files:
                next_name = "salidas/screenshots/captura_1.png"
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
                next_name = f"salidas/screenshots/captura_{next_num}.png"
                print(next_name)
            # Guardar la captura con el nombre generado
            cv2.imwrite(next_name, img)
            
def base_blanca():  # Cuando apretamos el boton ponemos el flag up, para poder seleccionar la base
    global gv, bb, primera
    bb=True
    primera = True
    gv.paused=True
    pausa.config(image = imagenBI)
    text4.config(text="Esperando area de detección")
    visualizar()

def base_blanca_aux(point1, point2): # Aca nada mas cargamos las variables y cambia el texto
    global bb, gv
    gv.y1=point1[1]
    gv.x1=point1[0]
    gv.y2=point2[1]
    gv.x2=point2[0]
    text4.config(text="Area de deteccion seleccionada ")
    text4.config(foreground='green')
    gv.yinicio=gv.y2 - 20
    gv.yfinal= gv.y1 + 20
    gv.paused=False
    
def habilitar_seleccion():
    global gv, seleccion_entrada_habilitada, primera
    primera = True
    pausa.config(image = imagenBI)
    seleccion_entrada_habilitada = not seleccion_entrada_habilitada
    gv.paused=True
    visualizar()

def eliminar_hojas(hojas, frame_actual):
    global gv # Hacer referencia a la variable global
    # Recorrer el array de hojas al revés para evitar problemas al eliminar elementos
    for i in range(len(hojas)-1, -1, -1):
        hoja = hojas[i]
        # Obtener la última aparición de la hoja
        primer_aparicion = hoja.apariciones[0]
        ultima_aparicion = hoja.apariciones[-1]
        # Si la distancia entre el frame de la última aparición y el frame actual es mayor a 15
        if frame_actual - ultima_aparicion.getframe() > gv.configuracion.getfpsapa():
            # Agregar la hoja al array de hojas perdidas
            if hoja.getcantapariciones()>gv.configuracion.getcantapa():
                if primer_aparicion.gety() > ultima_aparicion.gety():
                    gv.hojas_final.append(hoja)
                else:
                    gv.hojas_final_sale.append(hoja)
            # Eliminar la hoja del array hojas
            del hojas[i]
    return hojas
    
def finalizar():
    global gv
    gv.cap.release()
    cv2.destroyAllWindows()
    print("FIN")
    
def clicks():
    global gv, bb, seleccion_entrada_habilitada, click_count
    if click_count==2:
        if bb == True and seleccion_entrada_habilitada==False:    # Si el flag de que todavia no fue seleccionada la base es True, lo que hace es llamar a la funcion
            base_blanca_aux(gv.point1, gv.point2)
            bb = False # Despues cambiamos el flag para que los proximos dos puntos sean para seleccionar la conversion.
        elif seleccion_entrada_habilitada == True and bb == False:  # Selección de puntos de entrada y salida
            gv.entrada_coord = gv.point1
            gv.salida_coord = gv.point2
            gv.paused=False
            seleccion_entrada_habilitada = not seleccion_entrada_habilitada  # Cambiar bandera para evitar múltiples selecciones   
        else:
            # Calculate distance between two points
            distance = math.sqrt((gv.point2[0]-gv.point1[0])**2 + (gv.point2[1]-gv.point1[1])**2)
            # Display distance on GUI
            gv.cte=(10**2)/(distance**2)
            text2.config(text="Conv: "+ "%.2f" %gv.cte+"[mm^2/px^2]")
        # Reset click count and points
        click_count = 0
        gv.point1 = None
        gv.point2 = None
    return gv.paused

    

def escribirarchivo(hojas_final, hojas_final_sale, bandera):
    
    global gv
    gv.estadisticas = []
    if bandera == 0:
        #gv.archi1.write("header: ")
        gv.archi1.write(gv.configuracion.__json__())
        #gv.archi1.write("\n")
        for item in hojas_final:
            for aparicion in item.apariciones:
#                gv.archi1.write(str(item.id)+ "|"+str(aparicion.getx()) +"|"+ str(aparicion.gety()) +"|"+ 
#                             str(aparicion.getxp()) +"|"+ str(aparicion.getyp()) +"|"+str(aparicion.getarea())+"|"+ str(aparicion.getframe())+"\n")
                 gv.archi1.write('{"id":'+str(item.id)+',"x":'+str(aparicion.getx())+',"y":'+str(aparicion.gety())+',"xp":'+str(aparicion.getxp())+', "yp":'+str(aparicion.getyp())+',"area":'+str(aparicion.getarea())+',"frame":'+str(aparicion.getframe())+'},')
                        
        
    if bandera == 1:
        cont=0
        areaitem=0
        gv.archi2.seek(0, 2)
        
        # Guardamos en gv.estadisticas los valores estadisticos de cada area de la hoja siempre y cuando la hoja este entre el tiempo determinado
        gv.estadisticas = [item.getarea() for item in hojas_final if (gv.garch - gv.configuracion.gettiempo()) < (item.apariciones[0].getframe() / (30 * 60)) < gv.garch] 

        if len(gv.estadisticas) >0: #Si hay objetos en gv.estadisticas calculamos el promedio y lo guardamos.
            
            area_mediana = np.mean([item['mediana'] for item in gv.estadisticas])*gv.cte # Calcula el flujo promedio de areas em mm^2
            area_percentil25 = np.mean([item['percentil25'] for item in gv.estadisticas])*gv.cte
            area_percentil75 = np.mean([item['percentil75'] for item in gv.estadisticas])*gv.cte
            
            
            gv.archi2.write(str(len(gv.estadisticas))+"|"+str(area_mediana)+"|"+str(area_percentil25)+"|"+str(area_percentil75)+"|")
        gv.archi2.write(str(gv.garch-gv.configuracion.gettiempo())+"|"+str(gv.garch)+"\n")


    # archi1.write("Saliente: \n")
    # for item in hojas_final_sale:
    #     for aparicion in item.apariciones:
    #         archi1.write(str(item.id)+ "|"+str(aparicion.getx()) +"|"+ str(aparicion.gety()) +"|"+ str(aparicion.getarea())+"|"+str(aparicion.getframe())+"\n")

    

def visualizar():
        global gv
        if gv.cap is not None:
            global configuracion
            # Read a frame from the video
            gv.paused=clicks()
            if gv.paused==False:
                gv.frameactual+=1
                success, img = gv.cap.read()
                if success:
                    frame = img[gv.y1: gv.y2, gv.x1:gv.x2]
                    #dim=(640,640)
                    #frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                    # Run YOLOv8 inference on the frame
                    results = gv.model.predict(frame, conf=gv.configuracion.getconf())
                    # Visualize the results on the frame
                    #gv.annotated_frame = results[0].plot(show=True)
                    if gv.frameactual - gv.frameaux >= gv.configuracion.getfpsdist():
                        detector(results, gv.frameactual)
                        gv.frameaux=gv.frameactual
                    
                    sec=gv.frameactual/gv.configuracion.getfps()
                    s=datetime.timedelta(seconds=int(sec))
                    hora = gv.configuracion.gethora()+s
                    text6.config(text=str(hora))
                    
                    gv.garch = sec/60
                    
                    if (gv.garch % gv.configuracion.gettiempo()) == 0:
                        textdebug.config(text="Guarde " + str(gv.garch) + " veces")
                        escribirarchivo(gv.hojas_final, gv.hojas_final_sale, 1)
                        # escribimos el archivo del promedio de area en el intervalo de tiempo dado.
                        
                    #cv2.putText(img, "Hoja "+str(ID+1), (400, 50 - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255),2)  
                    gv.hojas = eliminar_hojas(gv.hojas, gv.frameactual)
                    
                    
                    text1.config(text="Hoja "+str(len(gv.hojas_final))+" "+str(gv.ID)+"\nSaliente: "+str(len(gv.hojas_final_sale)))

                    # area=calculararea(gv.hojas_final)
                    # area=area*gv.cte
                    # text3.config(text="Area:  "+"%.2f" %area + " [mm^2]")
                    
                    for i in range(len(gv.hojas_final)):
                        hoja=gv.hojas_final[i]
                        r = 0
                        g = 255
                        b = 0
                        for j in range(3, len(hoja.apariciones)-1):       #Dibujamos la trayectoria de hormiga y la trayectoria predicha
                            xi=hoja.apariciones[j].getx()+gv.x1
                            yi=hoja.apariciones[j].gety()+gv.y1
                            xf=hoja.apariciones[j+1].getx()+gv.x1
                            yf=hoja.apariciones[j+1].gety()+gv.y1
                            cv2.line(img,(xi,yi),(xf,yf),(r,g,b),1)
                   
                    
                    
                    cv2.rectangle(img, (gv.x1, gv.y1), (gv.x2, gv.y2), (0, 255, 0), 2)
                    cv2.line(img,(gv.x1,gv.yfinal),(gv.x2,gv.yfinal),(255,255,255),1)
                    cv2.line(img,(gv.x1,gv.yinicio),(gv.x2,gv.yinicio),(255,255,255),1)
                    cv2.circle(img, gv.entrada_coord, radius=5, color=(0, 255, 0), thickness=-1)  # (0, 255, 0) es verde en BGR
                    cv2.circle(img, gv.salida_coord, radius=5, color=(0, 0, 255), thickness=-1)  # (0, 0, 255) es rojo en BGR

                    
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                    # Convertimos el video
                    img = imutils.resize(img, width=640)
                    im = ImgPIL.fromarray(img)
                    img = ImageTk.PhotoImage(image=im)
                    
                    # img2 = cv2.cvtColor(gv.annotated_frame, cv2.COLOR_BGR2RGB)
                    # im2 = ImgPIL.fromarray(img2)
                    # img2 = ImageTk.PhotoImage(image=im2)

                    #cv2.imshow("YOLO", gv.annotated_frame)
                    
                    # Mostramos en el GUI
                    lblVideo.configure(image=img)
                    lblVideo.image = img
                    
                    
                    #if gv.frameactual >2:
                        #lblVideoYOLO.configure(image=img2)
                        #lblVideoYOLO.image = img2
                    
                    # Actualizar la barra de progreso
                    current_frame = gv.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    total_frames = gv.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    progress = current_frame / total_frames * 100
                    gv.progress_bar['value'] = progress
                    
                    lblVideoYOLO.config(highlightbackground='red', highlightthickness=2)
                   
                    
                    lblVideo.after(10, visualizar)
                else:
                    gv.cap.release()
                    if gv.filenames.index(gv.filename) == len(gv.filenames)-1:
                        text1.config(text="Hoja "+str(gv.ID+1)+"Finalizado")
                        escribirarchivo(gv.hojas_final, gv.hojas_final_sale, 0)
                        gv.archi1.write("]]")
                        gv.archi1.close()
                        gv.archi2.close()
                        cv2.destroyAllWindows()
                        #Exportar datos
                    else:
                        gv.filename=gv.filenames[gv.filenames.index(gv.filename)+1]
                        gv.cap = cv2.VideoCapture(gv.filename)
                        visualizar()

            

def crear_barra_progreso():
            global gv
            gv.progress_bar = ttk.Progressbar(pestania2, orient=HORIZONTAL, length=640, mode='determinate')
            gv.progress_bar.place(x=80, y=515)

def on_click(event):
            global gv, primera
            global click_count, point1, point2
            if click_count==0:
                gv.point1 = (event.x, event.y)
                click_count += 1
            elif click_count==1:
                gv.point2 = (event.x, event.y)
                click_count += 1
                if primera == True:
                    on_pause()
                    pausa.config(state= NORMAL)
                    primera = False
                



# Variables

gv.font=cv2.FONT_HERSHEY_SIMPLEX #Variable de texto que puede ser usada en opencv
gv.frameactual=0
gv.frameaux=0
gv.cte=0
bb=False
kf=[]
kf.append(KalmanFilter())
yfinalaux= 240
gv.ID=-1
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
seleccion_entrada_habilitada = False







#Pantalla principal
pantalla=Toplevel()
pantalla.title("GUI | TKINTER | HOJAS")
pantalla.geometry("1024x640")


#---------------------Configuracion de pestañas de la interfaz principal---------------------------
pestanias = Notebook(pantalla)

pestania1=Frame(pestanias)
pestania2=Frame(pestanias)


pestanias.add(pestania1, text="Init")
pestanias.add(pestania2, text="Pantalla Video")


pestanias.pack(expand=True, fill='both')


# Fondo
imagenF = PhotoImage(file="assets/Fondo.png")
background = Label(image = imagenF, text = "Fondo")
background.place(x = 0, y = 0, relwidth = 1, relheight = 1)



#---------------------Configuracion de botones de la interfaz principal---------------------------
# Iniciar Video
imagenBA = PhotoImage(file="assets/Abrir.png")
inicio = Button(pestania2, text="Iniciar",  command=iniciar)
inicio.place(x = 100, y = 580)

# Pausar/Reanudar Video
imagenBI = PhotoImage(file="assets/Inicio.png")
imagenBF = PhotoImage(file="assets/Finalizar.png")
pausa = Button(pestania2, text="Pausar",  command=on_pause)
pausa.place(x = 180, y = 580)
pausa.config(state="disabled")

# Capturar Frame
imagenBC = PhotoImage(file="assets/Capturar.png")
captura = Button(pestania2, text="Capturar",  command=capturar)
captura.place(x = 260, y = 580)
captura.config(state="disabled")

imagenBB = PhotoImage(file="assets/cuadrado.png")
base_b = Button(pestania2, text="Cuadrado",  command=base_blanca)
base_b.place(x = 340, y = 580)
base_b.config(state="disabled")

boton_seleccion = tk.Button(pestania2, text="Selección de Entrada y Salida", command=habilitar_seleccion)
boton_seleccion.place(x = 420, y = 580)



#---------------------Configuracion de textos de la interfaz principal---------------------------
text1 = Label(pestania2, text="Hoja "+str(gv.ID+1), font=("Cambria bold", 14))
text1.grid(row=0, column=1, padx=730, pady=(200,10), sticky="w")

text2 = Label(pestania2, text="Distancia de conversion: ", font=("Cambria bold", 14))
text2.grid(row=1, column=1, padx=730, pady=10, sticky="w")

text3 = Label(pestania2, text="Area ", font=("Cambria bold", 14))
text3.grid(row=2, column=1, padx=730, pady=10, sticky="w")

text4 = Label(pestania2, text="Seleccione el area de detección", font=("Cambria bold", 14))
text4.grid(row=3, column=1, padx=730, pady=10, sticky="w")
text4.config(foreground='red')

textdebug = Label(pestania2, text="Para debug", font=("Cambria bold", 14))
textdebug.grid(row=4, column=1, padx=730, pady=10, sticky="w")

text6 = Label(pestania2, text="", font =("Cambria bold", 12))
text6.place(x=720, y = 515)


#---------------------Configuracion del video en la interfaz---------------------------
lblVideo = Label(pestania2)
lblVideo.place(x = 80, y = 30)


lblVideoYOLO = Label(pestania2)
lblVideoYOLO.place(x = 730, y = 30)

pestanias.tab(1, state="disable")
crear_barra_progreso()


  

# Fecha
# Hora
# FPS
# Frames de distancia
# Cant frames sin aparicion
# Confianza
# Minima cantidad de apariciones
# Tiempo de guardado

def msgBox():
    
    msg.showerror('Error!', 'Error en los parametros de configuracion')
    
def crear_toolTip(widget, text):
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.show_tip(text)
    def leave(event):
        toolTip.hide_tip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

def guardar():      #Guardamos en el objeto configuracion los valores ingresados en las entradas
     global gv
     h, m = horastring.get().split(':')
     d = datetime.timedelta(hours=int(h), minutes=int(m))
     if fpstring.get() == 0 or fpsdisstring.get()== 0 or confstring.get()==0 or confstring.get()>1 or cantstring.get()==0 or tiemstring.get()==0:
         msgBox()
         return 0
     
     print(d)
     gv.configuracion = Configuracion(fechastring.get(), d, fpstring.get(),fpsdisstring.get(), fpsapastring.get(), confstring.get(), cantstring.get(), tiemstring.get())
     pestanias.tab(1, state="normal")
     pestanias.select(pestania2)


def validar_input(tipo, input):
    if tipo == 'int':
        return input.isdigit() or input == ""
    elif tipo == 'float':
        return input.replace('.', '', 1).isdigit() or input == ""    

def callback(tipo, P):
    return validar_input(tipo, P)



imagenQ = PhotoImage(file="assets/interrogatorio.png")


#Configuracion del grid
# Grid.rowconfigure(pestania1,0,weight=1)  #Configuramos el grid para ordenar los objetos dentro de la ventana
# Grid.columnconfigure(pestania1,0,weight=1)

# Grid.rowconfigure(pestania1,1,weight=1)

fechastring=tk.StringVar()      #Configuramos el texto variable de cada entrada y lo seteamos a un valor por defecto
fechastring.set("01-01-1970")

horastring=tk.StringVar()
horastring.set("00:00")

fpstring=tk.IntVar()
fpstring.set(30)

fpsdisstring=tk.IntVar()
fpsdisstring.set(2)

fpsapastring=tk.IntVar()
fpsapastring.set(15)

confstring=tk.DoubleVar()
confstring.set(0.6)

cantstring=tk.IntVar()
cantstring.set(10)

tiemstring=tk.DoubleVar()
tiemstring.set(10)

#Funciones callback para restringir los tipos de datos de cada entrada y evitar errores del usuario
vcmd_int = (pestania1.register(lambda P: callback('int', P)), '%P')
vcmd_float = (pestania1.register(lambda P: callback('float', P)), '%P')


fechat = Label(pestania1, text="Fecha:").grid(row=0, column=0, sticky=W) #Creamos el texto de cada entrada
fecha = Entry(pestania1, textvariable = fechastring, width=10).grid(row=0, column=1, sticky=W)   #Creamos la entrada de cada variable
fechaq = Button(pestania1, image= imagenQ, height="16", width="16", borderwidth=0)
fechaq.grid(row=0, column=2, sticky=W, padx=5)
crear_toolTip(fechaq, 'Fecha del video en formato DD-MM-YYYY')

horat = Label(pestania1, text="Hora:").grid(row=1, column=0, sticky=W)
hora = Entry(pestania1, textvariable = horastring, width=10).grid(row=1, column=1, sticky=W)
horaq = Button(pestania1, image= imagenQ, height="16", width="16", borderwidth=0)
horaq.grid(row=1, column=2, sticky=W, padx=5)
crear_toolTip(horaq, 'Hora de inicion del video en formato HH:MM')


fpst = Label(pestania1, text="FPS:").grid(row=2, column=0, sticky=W)
FPS = Entry(pestania1, textvariable = fpstring, width=10, validate='key', validatecommand=(vcmd_int)).grid(row=2, column=1, sticky=W)
fpsq = Button(pestania1, image= imagenQ, height="16", width="16", borderwidth=0)
fpsq.grid(row=2, column=2, sticky=W, padx=5)
crear_toolTip(fpsq, 'FPS del vídeo')


fpsdist = Label(pestania1, text="Distancia de Frames:").grid(row=3, column=0, sticky=W)
fpsdis = Entry(pestania1, textvariable = fpsdisstring, width=10, validate='key', validatecommand=(vcmd_int)).grid(row=3, column=1, sticky=W)
fpsdisq = Button(pestania1, image= imagenQ, height="16", width="16", borderwidth=0)
fpsdisq.grid(row=3, column=2, sticky=W, padx=5)
crear_toolTip(fpsdisq, 'Distancia entre frames de detección, cuanto mayor sea este numero\nmas rapido será el procesamiento a cambio de un mayor error')


fpsapat = Label(pestania1, text="Frames aparicion:").grid(row=4, column=0, sticky=W)
fpsapa = Entry(pestania1, textvariable = fpsapastring, width=10, validate='key', validatecommand=(vcmd_int)).grid(row=4, column=1, sticky=W)
fpsapaq = Button(pestania1, image= imagenQ, height="16", width="16", borderwidth=0)
fpsapaq.grid(row=4, column=2, sticky=W, padx=5)
crear_toolTip(fpsapaq, 'Cantidad de frames que deben pasar para dar por terminada una detección. Se recomienda no utilizar un valor mayor al de FPS')


conft = Label(pestania1, text="Confianza: ").grid(row=5, column=0, sticky=W)
conf = Entry(pestania1, textvariable = confstring, width=10, validate='key', validatecommand=(vcmd_float)).grid(row=5, column=1, sticky=W)
confq = Button(pestania1, image= imagenQ, height="16", width="16", borderwidth=0)
confq.grid(row=5, column=2, sticky=W, padx=5)
crear_toolTip(confq, 'Valor de umbral de confianza del modelo, entre 0 y 1, cuanto mayor sea el valor habrá menos falso positivos, pero se perderán detecciones')

cantapat = Label(pestania1, text="Cantidad de apariciones:").grid(row=6, column=0, sticky=W)
cantapa = Entry(pestania1, textvariable = cantstring, width=10, validate='key', validatecommand=(vcmd_int)).grid(row=6, column=1, sticky=W)
cantapaq = Button(pestania1, image= imagenQ, height="16", width="16", borderwidth=0)
cantapaq.grid(row=6, column=2, sticky=W, padx=5)
crear_toolTip(cantapaq, 'Cantidad de apariciones minimas necesarias para dar por positiva la completa detección')

tiemt = Label(pestania1, text="Tiempo de guardado:").grid(row=7, column=0, sticky=W)
tiem = Entry(pestania1, textvariable = tiemstring, width=10, validate='key', validatecommand=(vcmd_float)).grid(row=7, column=1, sticky=W)
tiemq = Button(pestania1, image= imagenQ, height="16", width="16", borderwidth=0)
tiemq.grid(row=7, column=2, sticky=W, padx=5)
crear_toolTip(tiemq, 'Intervalo de tiempo en minutos en el que se guardaron los datos procesados')

select = Label(pestania1, text="Carpeta de guardado").grid(row=8, column=0, sticky=W)
boton_seleccionar_carpeta = Button(pestania1, text="Seleccionar Carpeta", command=seleccionar_carpeta).grid(row=8, column=1, sticky=W)
selecq = Button(pestania1, image= imagenQ, height="16", width="16", borderwidth=0)
selecq.grid(row=8, column=2, sticky=W, padx=5)
crear_toolTip(selecq, 'Carpeta de guardado de los datos de procesamiento')


botonok = Button(pestania1, text="Confirmar", command=guardar)
botonok.grid(row=9, column=0, sticky=W)
botonok.config(state=DISABLED)

        
def quit_1():   #Funcion que cierra la ventana principal
    #finalizar()
    pantalla.destroy()
    pantalla.quit()
    #exit()
    

imagenS = PhotoImage(file="assets/salida.png")
salir = Button(pantalla, text="Salir", command=quit_1)
salir.place(x = 980, y = 600)

#Evento de click
lblVideo.bind("<Button-1>", on_click)

# Bucle de ejecucion de la ventana.
pantalla.mainloop()


# Release the video capture object and close the display window
gv.cap.release()
cv2.destroyAllWindows()
