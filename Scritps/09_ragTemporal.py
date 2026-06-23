# 09_ragTemporal.py
import os
import re
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from scipy.optimize import linear_sum_assignment
import networkx as nx
import pickle

# --- CONFIG ---
load_dotenv()

BASE    = os.path.dirname(os.getenv("CARPETA"))
CARPETA = os.path.join(BASE, "08_rootPersistence")
OUT_DIR = Path(os.path.join(BASE, "09_ragTemporal"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_AREA_NORM = 0.000001

# --- DETECCION AUTOMATICA DE RIZOTRONES (mismo patron que el script anterior) ---
PATRON = re.compile(r"^(.+)_DAS(\d+)$")

def agrupar_por_rizotron(carpeta: Path):
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
    print(f"Mask not found with the pattern '<prefix>_DAS<number>.png' in: {CARPETA}")
    exit()

print(f"Rhizotron detected ({len(rizotrones)}): {', '.join(rizotrones)}")

# --- FUNCIONES ---
def extraer_componentes(mask, das):
    H, W = mask.shape
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    comps = []
    for i in range(1, num_labels):
        area_px = stats[i, cv2.CC_STAT_AREA]
        area_norm = area_px / (H * W)
        if area_norm < MIN_AREA_NORM:
            continue

        cx_n  = centroids[i][0] / W
        cy_n  = centroids[i][1] / H
        y_top = stats[i, cv2.CC_STAT_TOP] / H
        y_bot = (stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) / H
        x_left  = stats[i, cv2.CC_STAT_LEFT] / W
        x_right = (stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]) / W

        comps.append({
            "id":        f"DAS{das}_L{i}",
            "das":       das,
            "label":     i,
            "area_px":   area_px,
            "area_norm": area_norm,
            "cx":        cx_n,
            "cy":        cy_n,
            "y_top":     y_top,
            "y_bot":     y_bot,
            "x_left":    x_left,
            "x_right":   x_right,
        })
    return comps, labels, H, W

def costo_arista(ca, cb):
    if cb["cy"] < ca["cy"] - 0.02:
        return 999

    t1 = abs(ca["area_px"] - cb["area_px"]) / (ca["area_px"] + cb["area_px"])
    t2 = abs(cb["das"] - ca["das"]) - 1

    dx = cb["cx"] - ca["cx"]
    dy = cb["cy"] - ca["cy"]
    dist = np.sqrt(dx**2 + dy**2)
    t3 = -dy / dist if dist > 0 else 0

    return t1 + t2 + t3

# --- PROCESAR CADA RIZOTRON ---
for prefijo in rizotrones:
    mascaras = grupos[prefijo]
    print(f"\n=== Rhizotron {prefijo} — {len(mascaras)} images ===")

    G = nx.DiGraph()
    serie = []

    for das, ruta in mascaras:
        mask = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
        comps, labels, H, W = extraer_componentes(mask, das)
        serie.append({"das": das, "comps": comps, "labels": labels, "H": H, "W": W})

        for c in comps:
            G.add_node(c["id"], **c)

        print(f"  DAS{das:>3}: {len(comps)} components")

    # --- CONECTAR DIAS CONSECUTIVOS CON ALGORITMO HUNGARO ---
    for i in range(len(serie) - 1):
        dia_a = serie[i]
        dia_b = serie[i + 1]
        ca_list = dia_a["comps"]
        cb_list = dia_b["comps"]

        if not ca_list or not cb_list:
            continue

        n_a, n_b = len(ca_list), len(cb_list)
        matriz = np.zeros((n_a, n_b))
        for ia, ca in enumerate(ca_list):
            for ib, cb in enumerate(cb_list):
                matriz[ia, ib] = costo_arista(ca, cb)

        filas, cols = linear_sum_assignment(matriz)
        UMBRAL = 2.0
        for f, c in zip(filas, cols):
            if matriz[f, c] <= UMBRAL:
                G.add_edge(ca_list[f]["id"], cb_list[c]["id"],
                          costo=round(matriz[f, c], 4))

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # --- IDENTIFICAR PIVOTANTE: componente mas arriba en primer DAS ---
    primer_das = serie[0]["das"]
    comps_primer_das = [c for c in serie[0]["comps"] if c["area_px"] >= 100]

    if not comps_primer_das:
        print(f"  WARNING: no valid components found in DAS{primer_das}, skipping {prefijo}")
        continue

    origen = min(comps_primer_das, key=lambda x: x["cy"])
    origen_id = origen["id"]
    print(f"  Pivot node: {origen_id} (area={origen['area_px']} px, cy={origen['cy']:.3f})")

    camino_pivotante = [origen_id]
    nodo_actual = origen_id
    cx_ref = G.nodes[origen_id]["cx"]

    for i in range(len(serie) - 1):
        das_siguiente = serie[i + 1]["das"]
        candidatos = [(n, d) for n, d in G.nodes(data=True)
                     if d["das"] == das_siguiente
                     and d["cy"] >= G.nodes[nodo_actual]["cy"] - 0.02
                     and abs(d["cx"] - cx_ref) <= 0.15]
        if not candidatos:
            continue
        mejor_nodo, mejor_data = max(candidatos, key=lambda x: x[1]["cy"])
        camino_pivotante.append(mejor_nodo)
        cx_ref = mejor_data["cx"]
        nodo_actual = mejor_nodo

    print(f"  Pivot path: {len(camino_pivotante)} nodes")

    # --- GUARDAR GRAFO Y CADENA PIVOTANTE ---
    out_grafo = OUT_DIR / f"{prefijo}_grafo.pkl"
    with open(out_grafo, "wb") as f:
        pickle.dump(G, f)

    out_pivotante = OUT_DIR / f"{prefijo}_pivotant.pkl"
    with open(out_pivotante, "wb") as f:
        pickle.dump(camino_pivotante, f)

    print(f"  saved: {out_grafo.name}, {out_pivotante.name}")

print(f"\nProcessing complete. Results in: {OUT_DIR}")