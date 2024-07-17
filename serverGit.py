import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Cargar el modelo al inicio
model = joblib.load('./models/githubN_GradientClass_best_model_0.68.pkl')

@app.route('/')
def home():
    return render_template('formGit.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del formulario
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])
        feature4 = float(request.form['feature4'])
        feature5 = float(request.form['feature5'])

        # Crear el array de entrada para la predicción
        X_test = np.array([feature1, feature2, feature3, feature4, feature5])

        # Realizar la predicción
        prediction = model.predict(X_test.reshape(1, -1))

        # Convertir la predicción a tipo serializable y asegurar que sea un entero
        prediction_value = int(prediction[0])

        # Determinar el resultado en base a la predicción
        if prediction_value == 0:
            result_message = "Toxicidad Baja"
        elif prediction_value == 1:
            result_message = "Toxicidad Media"
        elif prediction_value == 2:
            result_message = "Toxicidad Alta"
        else:
            result_message = "Toxicidad Desconocida"

        # Devolver la predicción y el mensaje como JSON
        return jsonify({
            'prediccion': prediction_value, 
            'mensaje': result_message,
            'valores_entrada': [feature1, feature2, feature3, feature4, feature5]
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    # Iniciar la aplicación Flask en el puerto 9001
    app.run(port=9001)
