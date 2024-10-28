
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carga los datos desde el archivo txt a un DataFrame
df = pd.read_csv('datos.txt', sep='|', header=None)
df.columns = ['ID', 'X', 'Y', 'Kalman_X', 'Kalman_Y', 'Area', 'Frame']

# Agrupa por ID de hoja y calcula las estadísticas deseadas
stats = df.groupby('ID')['Area'].agg(['mean',  
                                      lambda x: np.percentile(x, 25),  # Percentil 25
                                      lambda x: np.percentile(x, 50),  # Percentil 50 (esto es igual a la mediana)
                                      lambda x: np.percentile(x, 75)])  # Percentil 75


# Imprime las estadísticas
print(stats)

# Crea un gráfico de velas para todas las hojas
fig, ax = plt.subplots()
# Para cada hoja, añade una vela al gráfico
for id in df['ID'].unique():
    data = df[df['ID'] == id]['Area']
    bp = ax.boxplot(data, positions=[id], vert=True)

    # Agrega las etiquetas a las velas
    for element in ['boxes','means', 'caps']:
        plt.setp(bp[element], color='blue')
    for median in bp['medians']:
        median.set(color='red', linewidth=2)

ax.set_xlabel('ID de la hoja')
ax.set_ylabel('Área')
ax.set_title('Gráfico de velas para todas las hojas')
plt.show()
