#Escribir un programa que permita el ingreso de dos numeros y muestre por pantalla True si el primer numero es divisible para el segundo, False para caso contrario.
n1=int(input("Ingrese numero 1: "))
n2=int(input("Ingrese numero 2: "))

resto=n1%n2
logica=resto==0
print(logica)