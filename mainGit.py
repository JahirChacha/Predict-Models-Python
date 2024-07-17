import pandas as pd
from utilsGit import Utils
from modelsGit import Models

if __name__ == "__main__":
    utils = Utils()
    models = Models()

    # Cargar datos desde CSV
    print("Intentando cargar datos desde CSV...")
    try:
        data = utils.load_from_csv('./in/githubN.csv')
        print("Datos cargados correctamente.")
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        exit()

    # Imprimir los nombres de las columnas para verificar
    print("Columnas del dataset:", data.columns)
    
    # Verificar si '%Toxicos' está en las columnas
        # Trabajar con una muestra pequeña para pruebas rápidas
    if 'Toxicos' in data.columns:
        data_sample = data.sample(100, random_state=42)  # Ajustar el tamaño de la muestra según sea necesario
        print("Tamaño de muestra:", data_sample.shape)
        print(data_sample.head())  # Imprimir una muestra de los datos

        # Obtener características y objetivo
        features = ['Total commits', 'Total commits per day', 'Accumulated commits', 'Committers', 'Committers Weight']
        
        target = 'Toxicos'
        print("Características:", features)
        print("Objetivo:", target)
        
        print("Separando características y objetivo...")
        try:
            x, y = utils.features_target(data_sample, features, target)
            print("Características y objetivo separados correctamente.")
            print("Datos de entrada (X):")
            print(x.head())
            print("Objetivo (y):")
            print(y.head())
        except Exception as e:
            print(f"Error al separar características y objetivo: {e}")
            exit()

        print("Iniciando el entrenamiento del modelo...")
        try:
            best_model = models.grid_training(x, y, "githubN")
            print("Modelo entrenado correctamente.")
        except Exception as e:
            print(f"Error durante el entrenamiento del modelo: {e}")
            exit()

        print("Mejor modelo:", best_model)
    else:
        print("La columna '%Toxicos' no se encuentra en el dataset.")
