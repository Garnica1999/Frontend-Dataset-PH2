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
import cv2
import base64

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
# 2.5 LÓGICA DE GRAD-CAM
# ==============================================================================
def find_last_conv_layer(model):
    """Busca dinámicamente la última capa convolucional (salida 4D)."""
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
        # Si es un modelo anidado (backbone)
        elif isinstance(layer, tf.keras.Model):
            for inner_layer in reversed(layer.layers):
                if len(inner_layer.output_shape) == 4:
                    return layer.name, inner_layer.name
    return None

def make_gradcam_heatmap(img_array, model, pred_index=None):
    """Genera el mapa de calor Grad-CAM."""
    layer_info = find_last_conv_layer(model)
    if layer_info is None:
        return np.zeros((img_array.shape[1], img_array.shape[2]))

    try:
        if isinstance(layer_info, tuple):
            bb_name, conv_name = layer_info
            backbone = model.get_layer(bb_name)
            last_conv_layer = backbone.get_layer(conv_name)
            
            # Modelo que extrae la última conv y la salida del backbone
            grad_model = tf.keras.models.Model(
                [backbone.inputs], [last_conv_layer.output, backbone.output]
            )
            
            # Pasar por las capas previas al backbone
            x = img_array
            for layer in model.layers:
                if layer.name == bb_name:
                    break
                x = layer(x)
                
            with tf.GradientTape() as tape:
                conv_outputs, bb_outputs = grad_model(x)
                tape.watch(conv_outputs)
                
                # Pasar por las capas posteriores al backbone
                y = bb_outputs
                start_idx = model.layers.index(backbone) + 1
                for layer in model.layers[start_idx:]:
                    y = layer(y)
                
                preds = y
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return heatmap.numpy()
            
        else:
            grad_model = tf.keras.models.Model(
                [model.inputs], 
                [model.get_layer(layer_info).output, model.output]
            )

            with tf.GradientTape() as tape:
                last_conv_layer_output, preds = grad_model(img_array)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, last_conv_layer_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return heatmap.numpy()
    except Exception as e:
        import traceback
        print(f"Error al generar Grad-CAM: {e}")
        traceback.print_exc()
        return np.zeros((img_array.shape[1], img_array.shape[2]))

def overlay_gradcam(img_tensor, heatmap):
    """Superpone el heatmap sobre la imagen original."""
    img = img_tensor[0].numpy()
    # Asegurar que img esté en 0-255 uint8
    if np.max(img) <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    # Redimensionar heatmap al tamaño de la imagen
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superponer con 40% de opacidad para el heatmap
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # Convertir a base64
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

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

        # 3.6 Generar Grad-CAM
        try:
            heatmap = make_gradcam_heatmap(img_tensor, model, pred_index=idx_ganador)
            gradcam_base64 = overlay_gradcam(img_tensor, heatmap)
        except Exception as e:
            print(f"Error al generar Grad-CAM overlay: {e}")
            gradcam_base64 = None

        return JSONResponse(content={
            "is_cancer": is_cancer,
            "confidence": confidence,
            "predicted_class": clase_predicha,
            "probabilities": detalles,
            "filename": image.filename,
            "gradcam_base64": gradcam_base64
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