from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Habilitar CORS si es necesario para tu frontend

# Cargar el modelo de TensorFlow
modelo = tf.keras.models.load_model('/home/happodaikarin/proyectoRN/Backend/modelo/modeloV1')  # Asegúrate de ajustar la ruta

# Mapeo de índices de clase a etiquetas de aves
indice_a_etiqueta = {
    0: 'Especie1',
    1: 'Especie2',
    2: 'Especie3',
    3: 'Especie4',
    4: 'Especie5',
    5: 'Especie6',
    6: 'Especie7',
    7: 'Especie8',
    8: 'Especie9',
    9: 'Especie10',
    10: 'Especie11',
    11: 'Especie12',
    12: 'Especie13',
    13: 'Especie14',
    14: 'Especie15',
    15: 'Especie16',
    16: 'Especie17',
    17: 'Especie18',
    18: 'Especie19',
    19: 'Especie20',
    20: 'Especie21',
    21: 'Especie22',
    22: 'Especie23',
    23: 'Especie24',
    24: 'Especie25',
    25: 'Especie26',
    26: 'Especie27',
    27: 'Especie28',    
    28: 'Especie29'
    # Asegúrate de reemplazar 'EspecieX' con los nombres reales de las aves
}

def preparar_imagen(imagen, tamaño_objetivo):
    """Preprocesar la imagen para que sea adecuada para el modelo."""
    if imagen.mode != "RGB":
        imagen = imagen.convert("RGB")
    imagen = imagen.resize(tamaño_objetivo)
    imagen = tf.keras.preprocessing.image.img_to_array(imagen)
    imagen = np.expand_dims(imagen, axis=0)
    return imagen

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró el archivo'}), 400

    archivo = request.files['file']
    if archivo.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

    imagen = Image.open(io.BytesIO(archivo.read()))
    imagen_preparada = preparar_imagen(imagen, (224, 224))  # Ajustar según el tamaño de entrada del modelo
    predicciones = modelo.predict(imagen_preparada)
    clase_predicha = np.argmax(predicciones, axis=1)[0]  # Obtener la clase con mayor probabilidad

    etiqueta_predicha = indice_a_etiqueta.get(clase_predicha, 'Etiqueta no encontrada')  # Convertir índice en etiqueta

    return jsonify({'prediccion': etiqueta_predicha})

if __name__ == '__main__':
    app.run(debug=True)
