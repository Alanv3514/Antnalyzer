#Cree un programa que permita ingresar un número. Se deberá mostrar True si esta entre 10 y 100 (sin incluirlos), false en caso contrario.

n=int(input("Ingrese un número: "))
logica= not(n>10 and n<100)
print("¿El número no está entre 10 y 100: ", logica)