import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def crear_grafico_cajas(archivo):
    # Leer el archivo con pandas usando | como separador
    df = pd.read_csv(archivo, sep='|', decimal=',')
    
    # Ordenar el dataframe por hora (usando el nombre exacto con espacios)
    df = df.sort_values(' Hora ')
    
    # Crear los datos para el boxplot
    data = []
    labels = []
    
    for _, row in df.iterrows():
        # Crear array con los valores estadísticos usando los nombres exactos
        stats = [
            float(row[' Minimo ']), 
            float(row['Percentil 25']), 
            float(row['Mediana']), 
            float(row[' Percentil 75']), 
            float(row[' Maximo '])
        ]
        data.append(stats)
        labels.append(f'Hora {int(float(row[" Hora "]))}')

    # Configurar el estilo del gráfico
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Crear el boxplot
    bplot = ax.bxp(
        [{
            'whislo': stats[0],    # Mínimo
            'q1': stats[1],        # Q1
            'med': stats[2],       # Mediana
            'q3': stats[3],        # Q3
            'whishi': stats[4],    # Máximo
            'fliers': []           # Sin outliers
        } for stats in data],
        patch_artist=True          # Rellenar las cajas
    )

    # Personalizar el gráfico
    plt.title('Distribución por Hora', pad=20)
    plt.xlabel('Hora')
    plt.ylabel('Valor')
    
    # Colorear las cajas
    for box in bplot['boxes']:
        box.set_facecolor('lightblue')
        box.set_alpha(0.7)
    
    # Configurar las etiquetas del eje X
    plt.xticks(range(1, len(labels) + 1), labels)
    
    # Agregar grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Mostrar el gráfico
    plt.show()

# Ejecutar la función
crear_grafico_cajas('intervalo-13-11-2024.txt')
