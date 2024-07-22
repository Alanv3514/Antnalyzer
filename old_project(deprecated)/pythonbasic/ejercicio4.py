#Escriba un programa que permita al usuario ingresar radio(cm) y altura(cm) de un termo con forma cilindrica
#su programa debera calcular el volumen y mostrar True si puede llenarse con 300ml de agua.
#V=pi*R^2*H - 1ml=1cm^3

radio=float(input("Ingrese el radio del termo (en cm): "))
altura=float(input("Ingrese la altura del termo (en cm): "))
pi=3.14
vol=pi*(radio**2)*altura
print("El volumen del termo es: {:.2f} cm³".format(vol))
logica= vol>=300
print("¿Entran en el termo 300ml de agua?:", logica)
