%matplotlib qt

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
fps = 30
intervaloS= 600
# Convierte 'Frame' a minutos
df['Minute'] = df['Frame'] // (fps*intervaloS)

# Agrupa por 'Minute' y cuenta el número de hojas en cada grupo
counts = df.groupby('Minute')['ID'].nunique()

# Crea el gráfico
plt.figure()
plt.plot(counts)
plt.xlabel('Tiempo (minutos)')
plt.ylabel('Cantidad de hojas transportadas')
plt.title('Cantidad de hojas transportadas en intervalos de 1 minuto')
plt.show()
