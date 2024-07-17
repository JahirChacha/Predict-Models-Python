import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random

if __name__ == "__main__":
    # Leer el conjunto de datos
    dataset = pd.read_csv("./in/Clorofila y sombra.csv")
    
    # Simular la presencia de plagas en las plantas de pitahaya
    # Asumimos que la columna 'Tipo' indica si la planta es injertada o sin injertar
    # Generamos una columna 'Tipo' con valores aleatorios de ejemplo
    dataset['Tipo'] = ['Injertada' if i % 2 == 0 else 'Sin Injertar' for i in range(len(dataset))]
    
    # Generar una columna 'Plaga' basada en la categoría 'Tipo'
    # Suponemos que las plantas sin injertar tienen una mayor probabilidad de ser afectadas por plagas
    def asignar_plaga(tipo):
        if tipo == 'Sin Injertar':
            return random.choices([1, 0], weights=[0.7, 0.3])[0]
        else:
            return random.choices([1, 0], weights=[0.3, 0.7])[0]

    dataset['Plaga'] = dataset['Tipo'].apply(asignar_plaga)
    
    # Preparar los datos eliminando la columna 'Meses'
    X = dataset.drop(['Meses', 'Tipo'], axis=1)
    
    # Determinar el número óptimo de clusters utilizando el método del codo (opcional)
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 4))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Número de clusters')
    plt.ylabel('Distorsión')
    plt.title('Método del codo para encontrar el número óptimo de clusters')
    plt.show()
    
    # Aplicar el algoritmo K-Means con el número óptimo de clusters
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    
    # Añadir las etiquetas de los clusters al conjunto de datos original
    dataset['kmeans'] = kmeans.labels_
    
    # Imprimir el conjunto de datos con las etiquetas de clusters
    print("="*64)
    print(dataset)
    
    # Número de clusters encontrados
    print("="*64)
    print(f"Número de clusters encontrados: {kmeans.n_clusters}")
    
    # Centros de los clusters
    print("="*64)
    print("Centros de los clusters:")
    print(kmeans.cluster_centers_)
    
    # Análisis de frecuencia de plagas por tipo
    frecuencia_plagas = dataset.groupby('Tipo')['Plaga'].mean()
    print("="*64)
    print("Frecuencia de plagas por tipo:")
    print(frecuencia_plagas)
    
    # Exportar el conjunto de datos a un archivo Excel
    dataset.to_excel("./out/dataset_con_plaga.xlsx", index=False)
    print("El dataset con la nueva columna 'Plaga' se ha exportado a './out/dataset_con_plaga.xlsx'")
