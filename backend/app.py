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
def build_gradcam_components(model, conv_layer_name="conv5_block3_out"):
    """
    Desacopla el modelo cargado en 3 partes seguras (Preprocesamiento, Backbone y Cabeza)
    basado en la lógica comprobada del notebook de entrenamiento.
    """
    preprocess_layer = None
    base_resnet = None
    
    # 1. Extraer preprocesador y backbone
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "resnet" in layer.name.lower():
            base_resnet = layer
        elif hasattr(layer, "name") and "preprocess" in layer.name.lower():
            preprocess_layer = layer
            
    if base_resnet is None:
        raise ValueError("No se encontró sub-modelo ResNet en model.layers")

    # 2. Construir grad_model_base (desde el input de resnet hasta el output y la conv)
    conv_layer = base_resnet.get_layer(conv_layer_name)
    grad_model_base = tf.keras.models.Model(
        inputs=base_resnet.input,
        outputs=[conv_layer.output, base_resnet.output]
    )

    # 3. Construir head_model (las capas posteriores a ResNet50)
    resnet_out_shape = base_resnet.output_shape[1:]
    inp_head = tf.keras.Input(shape=resnet_out_shape)
    x = inp_head
    past_resnet = False
    
    for layer in model.layers:
        if layer is base_resnet:
            past_resnet = True
            continue
        if not past_resnet:
            continue
        x = layer(x) # Aquí sí funciona porque x e inp_head comparten la misma topología limpia
        
    head_model = tf.keras.models.Model(inputs=inp_head, outputs=x)

    return grad_model_base, preprocess_layer, head_model

def make_gradcam_heatmap(img_array, main_model, pred_index=None):
    """
    Genera el heatmap usando los 3 componentes desacoplados para evitar
    el error de positional arguments o disrupciones del GradientTape.
    """
    try:
        # Obtener los 3 componentes
        grad_model_base, preprocess_layer, head_model = build_gradcam_components(main_model)
        
        # 1. Preprocesamiento manual (fuera del Tape, ya que no tiene gradientes entrenables)
        if preprocess_layer is not None:
            arr_prep = preprocess_layer(img_array, training=False)
        else:
            arr_prep = tf.cast(img_array, tf.float32)

        # 2. Calcular Grad-CAM en una sola pasada limpia dentro del tape
        with tf.GradientTape() as tape:
            conv_outputs, resnet_out = grad_model_base(arr_prep, training=False)
            tape.watch(conv_outputs) # Vigilar el tensor interno explícitamente
            
            predictions = head_model(resnet_out, training=False)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # 3. Derivadas y promedios
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            print("Error: GradientTape no calculó gradientes (grads=None).")
            return np.zeros((img_array.shape[1], img_array.shape[2]))

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 4. Multiplicación de la matriz de activación por los pesos de importancia
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # 5. Normalización (incluimos el + 1e-7 de tu notebook para evitar división por cero)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-7)
        
        return heatmap.numpy()

    except Exception as e:
        import traceback
        print(f"Error interno al generar Grad-CAM en FastAPI: {e}")
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