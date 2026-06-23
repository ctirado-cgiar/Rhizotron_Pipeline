# 11b_angulos_laterales.py
import os
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from skimage.morphology import skeletonize
import pickle

# --- CONFIG ---
load_dotenv()

BASE         = os.path.dirname(os.getenv("CARPETA"))
CARPETA_MASK = os.path.join(BASE, "08_rootPersistence")
CARPETA_PIVOT = os.path.join(BASE, "10i_pivotanteReconstruido")
OUT_DIR      = Path(os.path.join(BASE, "11_rasgos"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANCHO_RIZOTRON_CM = 40
ALTO_RIZOTRON_CM  = 60
N_ZONAS           = 6
DAS_MAX           = 26
MIN_LARGO_LATERAL = 15  # pixeles minimos para considerar una lateral

# --- DETECCION DE RIZOTRONES ---
PATRON = re.compile(r"^(.+)_DAS(\d+)$")

def agrupar_por_rizotron(carpeta: Path):
    grupos = defaultdict(list)
    for ruta in carpeta.glob("*_DAS*.png"):
        m = PATRON.match(ruta.stem)
        if not m:
            continue
        prefijo, das_str = m.group(1), m.group(2)
        grupos[prefijo].append((int(das_str), ruta))
    def clave(p):
        m2 = re.match(r"^(\d+)_", p)
        return int(m2.group(1)) if m2 else float("inf")
    ordenados = sorted(grupos.keys(), key=clave)
    for p in ordenados:
        grupos[p].sort(key=lambda t: t[0])
    return ordenados, grupos

rizotrones, grupos = agrupar_por_rizotron(Path(CARPETA_MASK))
print(f"Rizotrones detectados ({len(rizotrones)}): {', '.join(rizotrones)}")

resultados = []

for prefijo in rizotrones:
    print(f"\n=== {prefijo} ===")

    linea_path = Path(CARPETA_PIVOT) / f"{prefijo}_linea_central.pkl"
    if not linea_path.exists():
        print(f"  AVISO: sin linea central, se omite")
        continue

    with open(linea_path, "rb") as f:
        linea_central = pickle.load(f)
    linea_arr = np.array(linea_central)

    # Usar mascara del DAS final <= DAS_MAX
    mascaras = [(d, r) for d, r in grupos[prefijo] if d <= DAS_MAX]
    if not mascaras:
        continue

    das_fin, ruta_fin = mascaras[-1]
    mask = cv2.imread(str(ruta_fin), cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    ESCALA_PX_CM = W / ANCHO_RIZOTRON_CM
    zona_alto_px = int(H / N_ZONAS)

    # Esqueleto completo
    skel = (skeletonize(binary // 255).astype(np.uint8)) * 255

    # Mascara del pivotante (dilatacion de la linea central)
    mask_piv = np.zeros((H, W), dtype=np.uint8)
    for r, c in linea_arr:
        if 0 <= int(r) < H and 0 <= int(c) < W:
            cv2.circle(mask_piv, (int(c), int(r)), 4, 255, -1)

    # Esqueleto de laterales = esqueleto - pivotante
    skel_lat = cv2.bitwise_and(
        skel,
        cv2.bitwise_not(cv2.dilate(mask_piv, np.ones((9,9), np.uint8)))
    )

    # Componentes de laterales
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        skel_lat, connectivity=8)

    partes   = prefijo.split("_", 1)
    plot     = partes[0]
    genotipo = partes[1] if len(partes) > 1 else prefijo

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_LARGO_LATERAL:
            continue

        pixeles = np.argwhere(labels == i)  # (r, c)

        # Punto mas cercano al pivotante = origen de la lateral
        dist_al_piv = np.array([
            np.min(np.sqrt(
                (linea_arr[:,0] - p[0])**2 +
                (linea_arr[:,1] - p[1])**2
            ))
            for p in pixeles
        ])
        idx_origen = np.argmin(dist_al_piv)
        origen     = pixeles[idx_origen]

        # Punto mas lejano del origen = punta de la lateral
        dists_desde_origen = np.sqrt(
            (pixeles[:,0] - origen[0])**2 +
            (pixeles[:,1] - origen[1])**2
        )
        idx_punta = np.argmax(dists_desde_origen)
        punta     = pixeles[idx_punta]

        # Longitud real de la lateral
        longitud_px = dists_desde_origen[idx_punta]
        if longitud_px < MIN_LARGO_LATERAL:
            continue
        longitud_cm = longitud_px / ESCALA_PX_CM

        # Vector origen -> punta
        dy = punta[0] - origen[0]  # positivo = hacia abajo
        dx = punta[1] - origen[1]  # positivo = hacia derecha

        # Angulo respecto a la vertical (0 = recto hacia abajo, 90 = horizontal)
        angulo_rad = np.arctan2(abs(dx), abs(dy))
        angulo_deg = np.degrees(angulo_rad)

        # Direccion: izquierda o derecha
        direccion = "right" if dx >= 0 else "left"

        # Zona de profundidad segun origen
        zona_idx  = min(int(origen[0] / zona_alto_px), N_ZONAS - 1)
        zona_cm0  = int(zona_idx * ALTO_RIZOTRON_CM / N_ZONAS)
        zona_cm1  = int((zona_idx + 1) * ALTO_RIZOTRON_CM / N_ZONAS)
        zona_name = f"{zona_cm0}_{zona_cm1}cm"

        resultados.append({
            "plot":        plot,
            "genotipo":    genotipo,
            "lateral_id":  i,
            "origen_r_px": int(origen[0]),
            "origen_c_px": int(origen[1]),
            "punta_r_px":  int(punta[0]),
            "punta_c_px":  int(punta[1]),
            "longitud_cm": round(longitud_cm, 3),
            "angulo_deg":  round(angulo_deg, 2),
            "direccion":   direccion,
            "zona":        zona_name,
        })

    n_lat = len([r for r in resultados if r["plot"] == plot])
    print(f"  Laterales procesadas: {n_lat}")

# --- GUARDAR ---
df_angulos = pd.DataFrame(resultados)
out_path   = OUT_DIR / "angulos_laterales.csv"
df_angulos.to_csv(out_path, index=False)
print(f"\nGuardado: {out_path} ({len(df_angulos)} laterales)")

# Resumen por genotipo y zona
resumen = df_angulos.groupby(["genotipo", "zona"]).agg(
    n_laterales    = ("lateral_id",  "count"),
    angulo_mean    = ("angulo_deg",  "mean"),
    angulo_std     = ("angulo_deg",  "std"),
    longitud_mean  = ("longitud_cm", "mean"),
    longitud_std   = ("longitud_cm", "std"),
).round(2).reset_index()

resumen_path = OUT_DIR / "angulos_laterales_resumen.csv"
resumen.to_csv(resumen_path, index=False)
print(f"Guardado: {resumen_path} ({len(resumen)} filas)")
print("\nListo.")