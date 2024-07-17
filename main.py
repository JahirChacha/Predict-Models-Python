import pandas as pd
from utils import Utils
from models import Models

if __name__ == "__main__":
    utils = Utils()
    models = Models()

    # Cargar datos desde CSV
    print("Intentando cargar datos desde CSV...")
    try:
        data = utils.load_from_csv('./in/dataset_con_plaga.csv')
        print("Datos cargados correctamente.")
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        exit()

    # Imprimir los nombres de las columnas para verificar
    print("Columnas del dataset:", data.columns)
    
    # Verificar si 'Plaga' está en las columnas
    if 'Plaga' in data.columns:
        # Trabajar con una muestra pequeña para pruebas rápidas
        data_sample = data.sample(100, random_state=42)  # Aumentar el tamaño de la muestra si es posible
        print("Tamaño de muestra:", data_sample.shape)
        print(data_sample.head())  # Imprimir una muestra de los datos

        # Obtener características y objetivo
        features = ['Meses', 'Descripcion', 'Tratamiento', 'Repeticion', 'Index', 'Bringht', 'Tipo', 'Ubicacion', 'kmeans']
        target = 'Plaga'
        
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
            best_model = models.grid_training(x, y, "dataset_con_plaga")
            print("Modelo entrenado correctamente.")
        except Exception as e:
            print(f"Error durante el entrenamiento del modelo: {e}")
            exit()

        print("Mejor modelo:", best_model)
    else:
        print("La columna 'Plaga' no se encuentra en el dataset.")
