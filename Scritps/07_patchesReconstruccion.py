# 07_patchesReconstruccion.py
import os, re
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

# ─── CONFIGURACIÓN ───────────────────────────────────────────────
RECONSTRUIR_MASK    = True
RECONSTRUIR_OVERLAY = True

BASE         = os.path.dirname(os.getenv("CARPETA"))
SRC_ORIG     = os.path.join(BASE, "03_dailySelection")
SRC_MASKS    = os.path.join(BASE, "06_rootSegmentation", "masks")
SRC_OVERLAYS = os.path.join(BASE, "06_rootSegmentation", "overlays")
DST_MASKS    = os.path.join(BASE, "07_patchesReconstruction", "mask")
DST_OVERLAYS = os.path.join(BASE, "07_patchesReconstruction", "overlay")
PATCH_SIZE   = int(os.getenv("PATCH_SIZE", 512))

if RECONSTRUIR_MASK:
    os.makedirs(DST_MASKS, exist_ok=True)
if RECONSTRUIR_OVERLAY:
    os.makedirs(DST_OVERLAYS, exist_ok=True)

# Patrón para extraer coordenadas reales: imagen__y<Y>_x<X>.png
PATRON = re.compile(r'^(.+)__y(\d+)_x(\d+)$')

# ─── AGRUPAR PATCHES POR IMAGEN ORIGINAL ─────────────────────────
patches_por_imagen = defaultdict(list)

src_lectura = SRC_MASKS if RECONSTRUIR_MASK else SRC_OVERLAYS

for f in os.listdir(src_lectura):
    if not f.endswith(".png"):
        continue
    stem = Path(f).stem
    m = PATRON.match(stem)
    if not m:
        print(f"  WARNING: wrong format name, skipping -> {f}")
        continue
    base, y, x = m.group(1), int(m.group(2)), int(m.group(3))
    patches_por_imagen[base].append((y, x, f))

print(f"Images to reconstruct: {len(patches_por_imagen)}")
#verificar que no falte ningún patch para las imágenes seleccionadas
def ya_existe(base):
    if RECONSTRUIR_MASK and not os.path.exists(os.path.join(DST_MASKS, base + ".png")):
        return False
    if RECONSTRUIR_OVERLAY and not os.path.exists(os.path.join(DST_OVERLAYS, base + ".png")):
        return False
    return True

pendientes = {b: p for b, p in patches_por_imagen.items() if not ya_existe(b)}
omitidas = len(patches_por_imagen) - len(pendientes)
if omitidas:
    print(f"  Skipped (already exist): {omitidas}")
print(f"Images pending to reconstruct: {len(pendientes)}")

patches_por_imagen = pendientes

# ─── RECONSTRUCCIÓN ───────────────────────────────────────────────
for base, patches in patches_por_imagen.items():
    print(f"  Reconstructing: {base}")

    # Leer dimensiones originales
    orig_path = None
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        candidate = os.path.join(SRC_ORIG, base + ext)
        if os.path.exists(candidate):
            orig_path = candidate
            break

    if orig_path is None:
        print(f"  WARNING: original image not found for {base}, skipping...")
        continue

    orig = cv2.imread(orig_path)
    H, W = orig.shape[:2]

    canvas_mask    = np.zeros((H, W), dtype=np.uint8)
    canvas_overlay = np.zeros((H, W, 3), dtype=np.uint8)

    for y, x, fname in patches:
        # Coordenadas reales tomadas directamente del nombre del archivo.
        # Ya no se calculan con r*step / c*step: cada patch (incluyendo
        # los de borde, que pueden tener solapamiento variable) trae
        # su posición exacta dentro de la imagen original.
        y2 = min(y + PATCH_SIZE, H)
        x2 = min(x + PATCH_SIZE, W)

        if RECONSTRUIR_MASK:
            mask_path = os.path.join(SRC_MASKS, fname)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    canvas_mask[y:y2, x:x2] = mask[:y2-y, :x2-x]

        if RECONSTRUIR_OVERLAY:
            overlay_path = os.path.join(SRC_OVERLAYS, fname)
            if os.path.exists(overlay_path):
                overlay = cv2.imread(overlay_path)
                if overlay is not None:
                    canvas_overlay[y:y2, x:x2] = overlay[:y2-y, :x2-x]

    if RECONSTRUIR_MASK:
        cv2.imwrite(os.path.join(DST_MASKS, base + ".png"), canvas_mask)

    if RECONSTRUIR_OVERLAY and np.any(canvas_overlay):
        cv2.imwrite(os.path.join(DST_OVERLAYS, base + ".png"), canvas_overlay)

    print(f"  OK: {base} ({W}x{H})")

print(f"\n{'='*60}")
print("Reconstruction completed!")
print(f"{'='*60}")