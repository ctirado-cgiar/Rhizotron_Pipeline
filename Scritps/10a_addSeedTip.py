# 10a_addSeedTip.py
import os
import re
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
import json

# --- CONFIG ---
load_dotenv()

BASE     = os.path.dirname(os.getenv("CARPETA"))
CARPETA  = os.path.join(BASE, "08_rootPersistence")
OUT_JSON = Path(BASE) / "seed_tip_coordinates.json"

TARGET_HEIGHT     = 900   # altura objetivo de la ventana en pixeles
DAS_MAX_PERMITIDO = int(os.getenv("DAS_MAX"))

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

rizotrones, grupos = agrupar_por_rizotron(Path(CARPETA))
if not rizotrones:
    print(f"Masks not found in: {CARPETA}")
    exit()
print(f"Rhizotrons detected ({len(rizotrones)}): {', '.join(rizotrones)}")

# --- CARGAR YA MARCADOS ---
datos = {}
if OUT_JSON.exists():
    with open(OUT_JSON, "r") as f:
        datos = json.load(f)
    print(f"Already marked: {list(datos.keys())}")

# --- VARIABLES GLOBALES ---
click_point = None

def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)

cv2.namedWindow("Point", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Point", mouse_callback)

def pedir_punto(img_color, prefijo, das_usado, etiqueta, color_punto):
    global click_point
    click_point = None
    print(f"{prefijo} (DAS{das_usado}) — click on {etiqueta}. 's'=confirm 'r'=reset 'q'=exit")

    while True:
        vis = img_color.copy()
        if click_point:
            cv2.circle(vis, click_point, 6, color_punto, -1)
            cv2.putText(vis, f"'s' to confirm {etiqueta}",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(vis, f"{prefijo}: click on {etiqueta}",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Point", vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('s') and click_point:
            return click_point
        elif key == ord('r'):
            click_point = None
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("Getting out...")
            exit()

# --- PROCESAR CADA RIZOTRON ---
for prefijo in rizotrones:
    if prefijo in datos:
        print(f"{prefijo} already marked, skipping...")
        continue

    mascaras = grupos[prefijo]
    mascaras_validas = [(d, r) for d, r in mascaras if d <= DAS_MAX_PERMITIDO]
    if not mascaras_validas:
        print(f"  WARNING: {prefijo} have not DAS <= {DAS_MAX_PERMITIDO}, will be skipped.")
        continue
    das_usado, ruta_usada = mascaras_validas[-1]

    mask = cv2.imread(str(ruta_usada), cv2.IMREAD_GRAYSCALE)
    H, W = mask.shape

    # Escala automatica segun altura objetivo
    SCALE = TARGET_HEIGHT / H
    W_display = int(W * SCALE)
    H_display = int(H * SCALE)

    img_display = cv2.resize(mask, (W_display, H_display),
                             interpolation=cv2.INTER_AREA)
    img_color   = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)

    print(f"\n{prefijo} | DAS{das_usado} | "
          f"Original: {W}x{H}px | Display: {W_display}x{H_display}px "
          f"| Scale: {SCALE:.3f}")

    # 1. Semilla (verde)
    pt_semilla = pedir_punto(img_color, prefijo, das_usado,
                             "the SEED (origin)", (0, 255, 0))
    img_color_marcada = img_color.copy()
    cv2.circle(img_color_marcada, pt_semilla, 6, (0, 255, 0), -1)

    # 2. Punta del pivotante (azul)
    pt_punta = pedir_punto(img_color_marcada, prefijo, das_usado,
                           "the TIP of the pivot", (255, 0, 0))

    # Convertir coordenadas display -> coordenadas reales -> normalizadas
    def a_norm(pt):
        x_real = int(pt[0] / SCALE)
        y_real = int(pt[1] / SCALE)
        return x_real, y_real, round(x_real / W, 4), round(y_real / H, 4)

    xs, ys, cxs, cys = a_norm(pt_semilla)
    xp, yp, cxp, cyp = a_norm(pt_punta)

    datos[prefijo] = {
        "das_referencia": das_usado,
        "W_ref":          W,
        "H_ref":          H,
        "scale_display":  round(SCALE, 4),
        "semilla": {"x_px": xs, "y_px": ys,
                    "cx_norm": cxs, "cy_norm": cys},
        "punta":   {"x_px": xp, "y_px": yp,
                    "cx_norm": cxp, "cy_norm": cyp},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(datos, f, indent=2)
    print(f"  Saving: {prefijo} DAS{das_usado} | "
          f"Seed=({xs},{ys}) tip=({xp},{yp})")

cv2.destroyAllWindows()
print(f"\nAll data saved to: {OUT_JSON}")