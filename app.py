from flask import Flask, request, render_template
import joblib
import numpy as np

# Cargar el modelo
modelo = joblib.load('modelo_titanic.pkl')  # Ajusta la ruta si tu modelo está en otra carpeta

# Crear la app
app = Flask(__name__)

# Ruta principal
@app.route('/')
def home():
    return render_template('index.html')

# Ruta de predicción
@app.route('/predecir', methods=['POST'])
def predecir():
    if request.method == 'POST':
        try:
            sexo = int(request.form['sexo'])
            clase = int(request.form['clase'])
            edad = float(request.form['edad'])
            sibsp = int(request.form['sibsp'])
            parch = int(request.form['parch'])
            fare = float(request.form['fare'])
            embarque = int(request.form['embarque'])

            if sexo not in [0, 1] or clase not in [1, 2, 3] or edad < 0 or sibsp < 0 or parch < 0 or fare < 0 or embarque not in [ 0, 1, 2]:
                return render_template('resultado.html', resultado="Error: Datos inválidos. Verifica tu información.")

            datos = np.array([[sexo, clase, edad, sibsp, parch, fare, embarque]])

            prediccion = modelo.predict(datos)
            probabilidad = modelo.predict_proba(datos)
            # Probabilidad de sobrevivir (índice 1)
            prob_survive = probabilidad[0][1] * 100
            prob_no_survive = probabilidad[0][0] * 100

            resultado = f'Probabilidad de sobrevivir: {prob_survive:.2f}%<br>Probabilidad de no sobrevivir: {prob_no_survive:.2f}%'

            #resultado = 'Sobreviviría' if prediccion[0] == 1 else 'No sobreviviría'

            return render_template('resultado.html', resultado=resultado)
        except Exception as e:
            # Si algo falla (ej: letra en vez de número), mostrar error
            return render_template('resultado.html', resultado="Error al procesar los datos. Intenta de nuevo.")

# IMPORTANTE: correr el servidor
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)