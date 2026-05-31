import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# Importar preprocesadores nativos necesarios para la capa custom
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mbn_preprocess

# ==============================================================================
# 1. REGISTRO DE LA CAPA CUSTOM (Crucial para poder cargar el modelo)
# ==============================================================================
@tf.keras.utils.register_keras_serializable(package='ph2')
class PreprocessLayer(tf.keras.layers.Layer):
    """Aplica el preprocess_input del backbone correspondiente."""
    def __init__(self, backbone='resnet', **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        
    def call(self, inputs):
        if self.backbone == 'efficientnet':
            return eff_preprocess(inputs)
        elif self.backbone == 'resnet':
            return resnet_preprocess(inputs)
        elif self.backbone == 'mobilenet':
            return mbn_preprocess(inputs)
        else:
            raise ValueError(f"Backbone desconocido: {self.backbone}")
            
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'backbone': self.backbone})
        return cfg

# ==============================================================================
# 2. CARGA DEL MODELO Y CONFIGURACIÓN GLOBAL
# ==============================================================================
app = FastAPI(title="Melanoma Detection API")

IMG_SIZE = (224, 224)
CLASS_NAMES = ['Common Nevus', 'Atypical Nevus', 'Melanoma']
SEED = 42

# Definimos las capas de Test-Time Augmentation (TTA)
tta_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
    tf.keras.layers.RandomRotation(0.1, seed=SEED),
])

print("Cargando modelo de Inteligencia Artificial (Exp 7)...")
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'modelo_exp7_ultra_regularizado.keras')

try:
    model = tf.keras.models.load_model(model_path)
    print("¡Modelo cargado exitosamente!")
except Exception as e:
    print(f"Error al cargar el modelo. Verifica la ruta: {model_path}")
    print(f"Detalle técnico: {e}")
    model = None

# ==============================================================================
# 3. ENDPOINT DE PREDICCIÓN CON TTA (EXPERIMENTO 13)
# ==============================================================================
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo proporcionado no es una imagen válida.")
    
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo de IA no está disponible en el servidor.")

    try:
        # 3.1 Leer los bytes de la imagen subida
        contents = await image.read()
        
        # 3.2 Decodificar la imagen a Tensor RGB y redimensionar
        # Usamos tf.io.decode_image que es robusto con jpg, png, bmp
        img_tensor = tf.io.decode_image(contents, channels=3, expand_animations=False)
        img_tensor = tf.image.resize(img_tensor, IMG_SIZE)
        img_tensor = tf.cast(img_tensor, tf.float32) # Dejar en rango [0, 255] float32
        
        # 3.3 Expandir dimensiones para simular el Batch (1, 224, 224, 3)
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        
        # 3.4 Lógica de Test-Time Augmentation (TTA) - 8 Augmentations
        n_augments = 8
        
        # Predicción base (sin alteración)
        proba_sum = model.predict(img_tensor, verbose=0)
        
        # Ciclo TTA para las variantes restantes
        for _ in range(n_augments - 1):
            aug_img = tta_aug(img_tensor, training=True) # training=True activa el giro/rotación
            proba_sum += model.predict(aug_img, verbose=0)
            
        # Promediar el acumulado de probabilidades
        proba_avg = proba_sum / n_augments
        probs_array = proba_avg[0]
        
        # 3.5 Determinar diagnóstico
        idx_ganador = np.argmax(probs_array)
        clase_predicha = CLASS_NAMES[idx_ganador]
        
        # Consideraremos "Cáncer" si la clase ganadora es Melanoma (Índice 2)
        is_cancer = bool(idx_ganador == 2)
        confidence = float(probs_array[idx_ganador])
        
        # Armamos un diccionario con el detalle de probabilidades para el frontend
        detalles = {clase: float(probs_array[i]) for i, clase in enumerate(CLASS_NAMES)}

        return JSONResponse(content={
            "is_cancer": is_cancer,
            "confidence": confidence,
            "predicted_class": clase_predicha,
            "probabilities": detalles,
            "filename": image.filename
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {str(e)}")

# ==============================================================================
# 4. SERVIR EL FRONTEND ESTÁTICO
# ==============================================================================
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"Advertencia: La carpeta del frontend no existe en {frontend_path}")