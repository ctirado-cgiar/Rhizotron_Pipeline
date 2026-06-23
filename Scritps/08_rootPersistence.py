# 08_rootPersistence.py
import os
import re
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# --- CONFIG ---
load_dotenv()

BASE    = os.path.dirname(os.getenv("CARPETA"))
CARPETA = os.path.join(BASE, "07_patchesReconstruccion", "mask")
OUT_DIR = Path(os.path.join(BASE, "08_rootPersistence"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- DETECCION AUTOMATICA DE RIZOTRONES ---
# Patron esperado: <id>_<GENOTIPO>_DAS<numero>.png  (ej: 5_HTA46_DAS31.png)
PATRON = re.compile(r"^(.+)_DAS(\d+)$")

def agrupar_por_rizotron(carpeta: Path):
    """Agrupa todos los .png de la carpeta por prefijo de rizotron (ej: '5_HTA46')."""
    grupos = defaultdict(list)
    for ruta in carpeta.glob("*_DAS*.png"):
        m = PATRON.match(ruta.stem)
        if not m:
            continue
        prefijo, das_str = m.group(1), m.group(2)
        grupos[prefijo].append((int(das_str), ruta))

    def clave_orden_rizotron(prefijo):
        m2 = re.match(r"^(\d+)_", prefijo)
        return int(m2.group(1)) if m2 else float("inf")

    rizotrones_ordenados = sorted(grupos.keys(), key=clave_orden_rizotron)
    for prefijo in rizotrones_ordenados:
        grupos[prefijo].sort(key=lambda t: t[0])

    return rizotrones_ordenados, grupos


rizotrones, grupos = agrupar_por_rizotron(Path(CARPETA))

if not rizotrones:
    print(f"No masks found with the pattern '<prefix>_DAS<number>.png' in: {CARPETA}")
    exit()

print(f"Rhizotron detected ({len(rizotrones)}): {', '.join(rizotrones)}")

for prefijo in rizotrones:
    mascaras = grupos[prefijo]
    print(f"\n=== Rhizotron {prefijo} — {len(mascaras)} images ===")

    # --- TAMANIO DE REFERENCIA: mediana de todas las imagenes de este rizotron ---
    altos  = [cv2.imread(str(r), cv2.IMREAD_GRAYSCALE).shape[0] for _, r in mascaras]
    anchos = [cv2.imread(str(r), cv2.IMREAD_GRAYSCALE).shape[1] for _, r in mascaras]
    H_ref  = int(np.median(altos))
    W_ref  = int(np.median(anchos))
    print(f"Reference size (median): {W_ref} x {H_ref} px")

    # --- ACUMULACION ---
    print(f"{'DAS':>5} {'New pixels':>15} {'Accumulated pixels':>20}")
    print("-" * 45)

    acumulada = np.zeros((H_ref, W_ref), dtype=np.uint8)

    for das, ruta in mascaras:
        mask = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Normalizar a coordenadas relativas y proyectar al tamanio de referencia
        # Usar INTER_NEAREST para no introducir grises en mascara binaria
        binary_ref = cv2.resize(binary, (W_ref, H_ref), interpolation=cv2.INTER_NEAREST)
        _, binary_ref = cv2.threshold(binary_ref, 127, 255, cv2.THRESH_BINARY)

        px_nuevos = np.sum(binary_ref > 0)
        acumulada = cv2.bitwise_or(acumulada, binary_ref)
        px_acum   = np.sum(acumulada > 0)

        # Mismo nombre que el archivo de entrada, sin sufijo
        out = OUT_DIR / ruta.name
        cv2.imwrite(str(out), acumulada)
        print(f"{das:>5} {px_nuevos:>15,} {px_acum:>20,}")

print(f"\nCommulative masks saved in: {OUT_DIR}")