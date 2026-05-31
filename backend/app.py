import os
import numpy as np
import cv2
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

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
# 2.5 LÓGICA DE GRAD-CAM (CORREGIDA PARA MODELOS ANIDADOS)
# ==============================================================================
def make_gradcam_heatmap(img_array, main_model, pred_index=None):
    """Genera el mapa de calor Grad-CAM con búsqueda robusta de la última capa convolucional."""
    
    # 1. Identificar el backbone anidado (ResNet50)
    backbone = None
    for layer in main_model.layers:
        if hasattr(layer, 'layers'): 
            backbone = layer
            break
            
    if backbone is None:
        print("No se detectó un modelo anidado (backbone).")
        return np.zeros((img_array.shape[1], img_array.shape[2]))

    # 2. Búsqueda SEGURA de la última capa convolucional
    last_conv_layer = None
    for layer in reversed(backbone.layers):
        # En lugar de consultar output_shape (que falla en capas Activation),
        # buscamos explícitamente el tipo de capa Conv2D. Es 100% seguro.
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
            
    if last_conv_layer is None:
        print("No se encontró una capa Conv2D en el backbone.")
        return np.zeros((img_array.shape[1], img_array.shape[2]))

    try:
        # 3. Crear el sub-modelo de gradientes
        grad_model = tf.keras.models.Model(
            inputs=backbone.inputs,
            outputs=[last_conv_layer.output, backbone.output]
        )
        
        with tf.GradientTape() as tape:
            # Pasar la imagen por las capas PREVIAS al backbone (Tu PreprocessLayer)
            x = img_array
            for layer in main_model.layers:
                if layer == backbone:
                    break
                x = layer(x)
                
            # Pasar el tensor por el backbone
            conv_outputs, backbone_outputs = grad_model(x)
            tape.watch(conv_outputs)
            
            # Pasar por las capas POSTERIORES (GlobalAvgPooling, Dense, Dropout...)
            y = backbone_outputs
            start_idx = main_model.layers.index(backbone) + 1
            for layer in main_model.layers[start_idx:]:
                y = layer(y)
                
            # Extraer la predicción final
            preds = y
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # 4. Calcular los gradientes reales
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 5. Construir el heatmap final
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()

    except Exception as e:
        import traceback
        print(f"Error interno al generar Grad-CAM: {e}")
        traceback.print_exc()
        return np.zeros((img_array.shape[1], img_array.shape[2]))

def overlay_gradcam(img_tensor, heatmap, original_filename):
    """Superpone el heatmap sobre la imagen original y la guarda en disco."""
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
    
    # Superponer con opacidad
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    now_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    safe_name = os.path.splitext(original_filename)[0].replace(" ", "_")
    filename = f'{now_str}_{safe_name}-gramcam.jpg'
    
    # Resolver ruta absoluta para guardar la imagen
    base_dir = os.path.dirname(os.path.dirname(__file__))
    gramcam_dir = os.path.join(base_dir, "gramcam")
    if not os.path.exists(gramcam_dir):
        os.makedirs(gramcam_dir)
        
    save_path = os.path.join(gramcam_dir, filename)
    cv2.imwrite(save_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    
    return f"/gramcam/{filename}"

# ==============================================================================
# 3. ENDPOINT DE PREDICCIÓN CON TTA E INTEGRACIÓN GRAD-CAM
# ==============================================================================
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo proporcionado no es una imagen válida.")
    
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo de IA no está disponible.")

    try:
        contents = await image.read()
        
        img_tensor = tf.io.decode_image(contents, channels=3, expand_animations=False)
        img_tensor = tf.image.resize(img_tensor, IMG_SIZE)
        img_tensor = tf.cast(img_tensor, tf.float32)
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        
        # Lógica TTA (8 Augmentations)
        n_augments = 8
        proba_sum = model.predict(img_tensor, verbose=0)
        
        for _ in range(n_augments - 1):
            aug_img = tta_aug(img_tensor, training=True)
            proba_sum += model.predict(aug_img, verbose=0)
            
        proba_avg = proba_sum / n_augments
        probs_array = proba_avg[0]
        
        idx_ganador = np.argmax(probs_array)
        clase_predicha = CLASS_NAMES[idx_ganador]
        is_cancer = bool(idx_ganador == 2)
        confidence = float(probs_array[idx_ganador])
        
        detalles = {clase: float(probs_array[i]) for i, clase in enumerate(CLASS_NAMES)}

        # Generar Grad-CAM basado en el diagnóstico principal
        try:
            heatmap = make_gradcam_heatmap(img_tensor, model, pred_index=idx_ganador)
            gradcam_url = overlay_gradcam(img_tensor, heatmap, image.filename)
        except Exception as e:
            print(f"Error generando superposición: {e}")
            gradcam_url = None

        return JSONResponse(content={
            "is_cancer": is_cancer,
            "confidence": confidence,
            "predicted_class": clase_predicha,
            "probabilities": detalles,
            "filename": image.filename,
            "gradcam_url": gradcam_url
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {str(e)}")

# ==============================================================================
# 4. SERVIR ARCHIVOS ESTÁTICOS Y FRONTEND
# ==============================================================================
base_dir = os.path.dirname(os.path.dirname(__file__))
gramcam_dir = os.path.join(base_dir, "gramcam")

if not os.path.exists(gramcam_dir):
    os.makedirs(gramcam_dir)

app.mount("/gramcam", StaticFiles(directory=gramcam_dir), name="gramcam")

frontend_path = os.path.join(base_dir, "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"Advertencia: La carpeta del frontend no existe en {frontend_path}")