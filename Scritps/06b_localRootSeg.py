#06b_localRootSeg.py
"""
Script para segmentación semántica de raíces en imágenes de rizotrones.
Modelo: Roboflow 2.0 Semantic Segmentation (MODEL_PATH)
Importante: Este scritp funciona si tiene el modelo descargado y guardado en la carpeta de modelos locales.
Genera:
- Máscaras binarias PNG (mismo nombre que original)
- Overlays visuales (imagen + máscara coloreada)
"""
#Por escribir la logica.
from inference import get_model
import cv2
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv
import base64
from PIL import Image
import io

# ─── CONFIG ──────────────────────────────────────────────────────
load_dotenv()

MODEL_PATH        = str(os.getenv("MODEL_PATH"))
BASE            = os.path.dirname(os.getenv("CARPETA"))
INPUT_FOLDER    = os.path.join(BASE, "05_patchesGeneration")
OUTPUT_FOLDER   = os.path.join(BASE, "06_rootSegmentation")
CONFIDENCE      = 1

OUT_MASKS    = os.path.join(OUTPUT_FOLDER, "masks")
OUT_OVERLAYS = os.path.join(OUTPUT_FOLDER, "overlays")

for folder in [OUT_MASKS, OUT_OVERLAYS]:
    Path(folder).mkdir(parents=True, exist_ok=True)

# ─── CARGAR MODELO ───────────────────────────────────────────────
print(f"Cargando modelo: {MODEL_PATH}...")
model = get_model(model_id=MODEL_PATH)

# ─── IMÁGENES A PROCESAR ─────────────────────────────────────────
extensiones_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
'''
#Lo comento para ejecutar solo las faltantes y no volver a procesar las que ya tienen mascara y overlay
imagenes = [f for f in os.listdir(INPUT_FOLDER)
            if Path(f).suffix.lower() in extensiones_validas]
'''
#Lo siguiente ejecuta solo las faltantes, es decir, las que no tienen mascara ni overlay
imagenes = []

for f in os.listdir(INPUT_FOLDER):
    if Path(f).suffix.lower() not in extensiones_validas:
        continue

    mask_path = os.path.join(OUT_MASKS, Path(f).stem + ".png")

    # si ya existe máscara, se asume procesada
    if os.path.exists(mask_path):
        continue

    imagenes.append(f)
#Hasta aquí, 'imagenes' contiene solo los archivos que no tienen máscara generada, evitando reprocesar los ya hechos.
print(f"\n{'='*60}")
print(f"Encontradas {len(imagenes)} imágenes para procesar")
print(f"{'='*60}\n")

# ─── LOOP PRINCIPAL ──────────────────────────────────────────────
for idx, imagen_nombre in enumerate(imagenes, 1):
    print(f"[{idx}/{len(imagenes)}] Procesando: {imagen_nombre}")

    ruta_imagen = os.path.join(INPUT_FOLDER, imagen_nombre)
    image = cv2.imread(ruta_imagen)

    if image is None:
        print(f"  ⚠️  Error al leer {imagen_nombre}, saltando...")
        continue

    h, w = image.shape[:2]

    try:
        # ── Inferencia ──────────────────────────────────────────
        results = model.infer(image, confidence=CONFIDENCE / 100)[0]
        # ── Decodificar segmentation_mask desde base64 ──────────
        mask_binaria = np.zeros((h, w), dtype=np.uint8)

        pred = results.predictions
        if hasattr(pred, 'segmentation_mask') and pred.segmentation_mask:
            # Decodificar PNG base64 → array numpy
            mask_bytes = base64.b64decode(pred.segmentation_mask)
            mask_pil = Image.open(io.BytesIO(mask_bytes)).convert('L')  # grayscale
            mask_arr = np.array(mask_pil)  # valores: 0=background, 1=Raiz

            # Escalar al tamaño original si difiere
            if mask_arr.shape != (h, w):
                mask_pil_resized = mask_pil.resize((w, h), Image.NEAREST)
                mask_arr = np.array(mask_pil_resized)

            # Clase 1 = Raiz → 255 en máscara binaria
            mask_binaria[mask_arr == 1] = 255

        # ── Guardar máscara (mismo nombre, extensión .png) ──────
        mask_path = os.path.join(OUT_MASKS, Path(imagen_nombre).stem + ".png")
        cv2.imwrite(mask_path, mask_binaria)

        # ── Guardar overlay (mismo nombre, extensión .png) ──────
        overlay = image.copy()
        overlay[mask_binaria == 255] = (0, 200, 60)
        blended = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)
        overlay_path = os.path.join(OUT_OVERLAYS, Path(imagen_nombre).stem + ".png")
        cv2.imwrite(overlay_path, blended)

        pixels_raiz = int(np.sum(mask_binaria == 255))
        print(f"  ✓ Máscara guardada | píxeles raíz: {pixels_raiz}")

    except Exception as e:
        print(f"  ✗ Error en inferencia: {e}")
        continue

print(f"\n{'='*60}")
print("✓ Proceso completado!")
print(f"  Masks    → {OUT_MASKS}")
print(f"  Overlays → {OUT_OVERLAYS}")
print(f"{'='*60}")
