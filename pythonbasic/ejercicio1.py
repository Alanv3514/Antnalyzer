#Elabore un programa en python que pida al usuario una cantidad de segundos y muestre el resultados en horas minutos y segundos.

segundos=int(input("Ingrese la cantidad de segundos: "))
#Primero lo que haremos es ver la cantidad de minutos que hay en los segundos, para esto utilizaremos el modulo
minutos=segundos//60
segundos=segundos%60
horas=minutos//60
minutos=minutos%60
print("Horas: ", horas, "Minutos: ", minutos, "Segundos: ", segundos)
