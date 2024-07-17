import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Cargar el modelo al inicio
model = joblib.load('./models/dataset_con_plaga_GradientClass_best_model_0.76.pkl')

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del formulario
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])
        feature4 = float(request.form['feature4'])
        feature5 = float(request.form['feature5'])
        feature6 = float(request.form['feature6'])
        feature7 = float(request.form['feature7'])
        feature8 = float(request.form['feature8'])
        feature9 = float(request.form['feature9'])

        # Crear el array de entrada para la predicción
        X_test = np.array([feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9])

        # Realizar la predicción
        prediction = model.predict(X_test.reshape(1, -1))

        # Redondear la predicción a 2 decimales y convertir a tipo serializable
        rounded_prediction = round(float(prediction[0]), 2)

        # Determinar el resultado en base a la predicción
        if rounded_prediction == 0:
            result_message = "Pitahaya sin Plaga"
        else:
            result_message = "Pitahaya con Plaga"

        # Devolver la predicción y el mensaje como JSON
        return jsonify({'prediccion': rounded_prediction, 'mensaje': result_message})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    # Iniciar la aplicación Flask en el puerto 9001
    app.run(port=9001)