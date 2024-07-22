iva=0.12
producto1="Capuchino"
valor1=2.60
total1=valor1 + (valor1*iva)
producto2="Expresso" 
valor2=1.10
total2=valor2 + (valor2*iva)
producto3="Mocaccino" 
valor3=2.95 
total3=valor3 + (valor3*iva)
print("{:^20}{:^20}{:^20}".format("PRODUCTO","VALOR SIN IVA","TOTAL"))
print("{:^20}{:^20,.2f}{:^20,.2f}".format(producto1,valor1,total1))
print("{:^20}{:^20,.2f}{:^20,.2f}".format(producto2,valor2,total2))
print("{:^20}{:^20,.2f}{:^20,.2f}".format(producto3,valor3,total3))