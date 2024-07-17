import pandas as pd
import joblib

class Utils:
    def load_from_csv(self, path):
        """Cargar datos desde un archivo CSV."""
        print(f"Cargando datos desde: {path}")
        try:
            data = pd.read_csv(path)
            print(f"Datos cargados: {data.shape}")
            return data
        except Exception as e:
            print(f"Error cargando datos desde el archivo CSV: {e}")
            raise

    def load_from_mysql(self):
        """Este método puede ser implementado para cargar datos desde MySQL."""
        pass

    def features_target(self, dataset, feature_cols, target_col):
        """
        Separar las características y el objetivo del conjunto de datos.

        Parameters:
        - dataset: DataFrame que contiene los datos.
        - feature_cols: Columnas a usar como características.
        - target_col: Columna objetivo.

        Returns:
        - X: DataFrame de características.
        - y: Serie objetivo.
        """
        print(f"Extrayendo características: {feature_cols} y objetivo: {target_col}")
        try:
            X = dataset[feature_cols]
            y = dataset[target_col]
            print(f"Tamaño de características (X): {X.shape}")
            print(f"Tamaño del objetivo (y): {y.shape}")
            return X, y
        except Exception as e:
            print(f"Error separando características y objetivo: {e}")
            raise

    def model_export(self, clf, score, model_name):
        """
        Exportar el modelo entrenado a un archivo.

        Parameters:
        - clf: Modelo entrenado.
        - score: Puntaje del modelo.
        - model_name: Nombre del modelo.
        """
        filename = f'./models/{model_name}_best_model_{round(score, 3)}.pkl'
        try:
            joblib.dump(clf, filename)
            print(f"Modelo exportado: {filename}")
        except Exception as e:
            print(f"Error exportando el modelo: {e}")
            raise
