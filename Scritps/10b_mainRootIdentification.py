# 10b_mainRootIdentification.py
import os
import re
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from scipy.ndimage import distance_transform_edt
import heapq
import json
import pickle

# --- CONFIG ---
load_dotenv()

BASE         = os.path.dirname(os.getenv("CARPETA"))
CARPETA_MASK = os.path.join(BASE, "08_rootPersistence")
PUNTOS_JSON  = Path(BASE) / "seed_tip_coordinates.json"
OUT_DIR      = Path(os.path.join(BASE, "10b_mainRootIdentification"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

PASO              = 4
SUAVIZADO_VENTANA = 11
DAS_MAX           = int(os.getenv("DAS_MAX", 100))
RDP_EPSILON       = 20
SCORE_MIN_ALIN    = 0.3
TARGET_HEIGHT_VIS = 900  # altura de visualizacion en pixeles

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
if not rizotrones:
    print(f"Masks not found in: {CARPETA_MASK}")
    exit()

if not PUNTOS_JSON.exists():
    print(f"File not found: {PUNTOS_JSON}. Run 10a_addSeedTip.py first.")
    exit()

with open(PUNTOS_JSON, "r") as f:
    puntos = json.load(f)

print(f"Rhizotrons detected ({len(rizotrones)}): {', '.join(rizotrones)}")

# --- FUNCIONES ---
def dijkstra_dt(dist_map, start, end, H, W):
    INF = float('inf')
    dist_arr = np.full((H, W), INF)
    dist_arr[start] = 0
    heap = [(0.0, start)]
    prev = {}
    vecinos_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    while heap:
        costo_actual, (r, c) = heapq.heappop(heap)
        if (r, c) == end:
            break
        if costo_actual > dist_arr[r, c]:
            continue
        for dr, dc in vecinos_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            d_eucl   = np.sqrt(dr**2 + dc**2)
            d_blanco = max(dist_map[nr, nc], 0.1)
            nuevo_costo = costo_actual + d_eucl / d_blanco
            if nuevo_costo < dist_arr[nr, nc]:
                dist_arr[nr, nc] = nuevo_costo
                prev[(nr, nc)] = (r, c)
                heapq.heappush(heap, (nuevo_costo, (nr, nc)))

    if end not in prev:
        n = max(int(np.hypot(end[0]-start[0], end[1]-start[1])), 1)
        return [(int(round(start[0]+(end[0]-start[0])*i/n)),
                 int(round(start[1]+(end[1]-start[1])*i/n)))
                for i in range(n+1)]

    camino = []
    nodo = end
    while nodo in prev:
        camino.append(nodo)
        nodo = prev[nodo]
    camino.append(start)
    camino.reverse()
    return camino

def punto_mejor_score(binary, semilla_px, punta_px, H, W):
    eje = np.array([punta_px[0]-semilla_px[0],
                    punta_px[1]-semilla_px[1]], dtype=float)
    norma_eje = np.linalg.norm(eje)
    if norma_eje > 1e-6:
        eje /= norma_eje

    r0, c0 = semilla_px
    if binary[r0, c0] == 0:
        ys, xs = np.where(binary > 0)
        if len(ys) == 0:
            return semilla_px
        dists = (ys-r0)**2 + (xs-c0)**2
        idx = np.argmin(dists)
        r0, c0 = int(ys[idx]), int(xs[idx])

    visitado = np.zeros((H, W), dtype=bool)
    visitado[r0, c0] = True
    cola = [(r0, c0)]
    mejor_punto = (r0, c0)
    mejor_score = 0.0
    vecinos_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    while cola:
        nuevos = []
        for r, c in cola:
            for dr, dc in vecinos_8:
                nr, nc = r+dr, c+dc
                if (0 <= nr < H and 0 <= nc < W and
                        not visitado[nr, nc] and binary[nr, nc] > 0):
                    visitado[nr, nc] = True
                    nuevos.append((nr, nc))
                    v = np.array([nr-semilla_px[0],
                                  nc-semilla_px[1]], dtype=float)
                    dist = np.linalg.norm(v)
                    if dist < 1e-6:
                        continue
                    alin = float(np.dot(v/dist, eje))
                    if alin >= SCORE_MIN_ALIN:
                        score = dist * alin
                        if score > mejor_score:
                            mejor_score = score
                            mejor_punto = (nr, nc)
        cola = nuevos

    return mejor_punto

def rdp(puntos, epsilon):
    if len(puntos) < 3:
        return puntos
    p0 = np.array(puntos[0], dtype=float)
    p1 = np.array(puntos[-1], dtype=float)
    linea = p1 - p0
    norma = np.linalg.norm(linea)
    if norma < 1e-6:
        dists = [np.linalg.norm(np.array(p, dtype=float) - p0)
                 for p in puntos[1:-1]]
    else:
        linea_n = linea / norma
        dists = []
        for p in puntos[1:-1]:
            v = np.array(p, dtype=float) - p0
            proj = np.dot(v, linea_n)
            perp = v - proj * linea_n
            dists.append(np.linalg.norm(perp))
    idx_max  = int(np.argmax(dists)) + 1
    dist_max = dists[idx_max - 1]
    if dist_max > epsilon:
        izq = rdp(puntos[:idx_max+1], epsilon)
        der = rdp(puntos[idx_max:], epsilon)
        return izq[:-1] + der
    else:
        return [puntos[0], puntos[-1]]

def eliminar_retrocesos(ruta, origen, punta):
    eje = np.array([punta[0]-origen[0], punta[1]-origen[1]], dtype=float)
    norma = np.linalg.norm(eje)
    if norma < 1e-6:
        return ruta
    eje /= norma

    def proj(p):
        return float(np.dot(np.array([p[0]-origen[0],
                                       p[1]-origen[1]], float), eje))

    ruta_limpia = list(ruta)
    for _ in range(30):
        proyecciones = [proj(p) for p in ruta_limpia]
        idx_inicio   = None
        proj_max     = proyecciones[0]
        for i in range(1, len(ruta_limpia)):
            if proyecciones[i] < proj_max - 2.0:
                idx_inicio = i - 1
                break
            proj_max = max(proj_max, proyecciones[i])
        if idx_inicio is None:
            break
        proj_inicio = proyecciones[idx_inicio]
        idx_fin = None
        for j in range(idx_inicio+1, len(ruta_limpia)):
            if proyecciones[j] >= proj_inicio:
                idx_fin = j
                break
        if idx_fin is None:
            ruta_limpia = ruta_limpia[:idx_inicio+1]
            ruta_limpia.append(ruta[-1])
            break
        p0 = ruta_limpia[idx_inicio]
        p1 = ruta_limpia[idx_fin]
        n  = max(int(np.hypot(p1[0]-p0[0], p1[1]-p0[1])), 1)
        puente = [(int(round(p0[0]+(p1[0]-p0[0])*i/n)),
                   int(round(p0[1]+(p1[1]-p0[1])*i/n)))
                  for i in range(n+1)]
        ruta_limpia = ruta_limpia[:idx_inicio] + puente + ruta_limpia[idx_fin+1:]
    return ruta_limpia

def suavizar_ruta(ruta, ventana=11):
    if len(ruta) < ventana * 2:
        return ruta
    arr      = np.array(ruta, dtype=float)
    suavizada = arr.copy()
    half     = ventana // 2
    for i in range(half, len(arr) - half):
        suavizada[i] = arr[i-half:i+half+1].mean(axis=0)
    suavizada[0]  = arr[0]
    suavizada[-1] = arr[-1]
    return [tuple(map(int, p)) for p in suavizada]

# --- PROCESAR CADA RIZOTRON ---
for prefijo in rizotrones:
    print(f"\n=== Rhizotron {prefijo} ===")

    if prefijo not in puntos:
        print(f"  WARNING: no seed/tip marked, skipping...")
        continue

    p_json   = puntos[prefijo]
    mascaras = grupos[prefijo]
    mascaras_validas = [(d, r) for d, r in mascaras if d <= DAS_MAX]
    if not mascaras_validas:
        print(f"  WARNING: no valid masks found")
        continue

    # Correccion de nombres: compatibilidad con seed_tip_coordinates.json
    semilla_px = (p_json["semilla"]["y_px"], p_json["semilla"]["x_px"])
    punta_px   = (p_json["punta"]["y_px"],   p_json["punta"]["x_px"])

    mask_ref = cv2.imread(str(mascaras_validas[-1][1]), cv2.IMREAD_GRAYSCALE)
    H_ref, W_ref = mask_ref.shape

    # --- FASE 1: Waypoints con score de alineacion ---
    print(f"  Phase 1: aligned waypoints ({len(mascaras_validas)} DAS)...")
    waypoints    = [semilla_px]
    dist_max_ant = 0

    for das, ruta_mask in mascaras_validas:
        mask = cv2.imread(str(ruta_mask), cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if binary.shape != (H_ref, W_ref):
            binary = cv2.resize(binary, (W_ref, H_ref),
                                interpolation=cv2.INTER_NEAREST)
            _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

        punta_das = punto_mejor_score(binary, semilla_px, punta_px, H_ref, W_ref)
        dist_das  = np.hypot(punta_das[0]-semilla_px[0],
                             punta_das[1]-semilla_px[1])

        if dist_das > dist_max_ant * 0.95:
            dist_max_ant = dist_das
            ultimo_wp    = waypoints[-1]
            if np.hypot(punta_das[0]-ultimo_wp[0],
                        punta_das[1]-ultimo_wp[1]) > PASO * 3:
                waypoints.append(punta_das)

    waypoints.append(punta_px)
    print(f"  Waypoints: {len(waypoints)}")
    for i, wp in enumerate(waypoints):
        d = np.hypot(wp[0]-semilla_px[0], wp[1]-semilla_px[1])
        print(f"    WP{i}: {wp} dist={d:.0f}px")

    # --- FASE 2: Dijkstra entre waypoints consecutivos ---
    print(f"  Phase 2: Dijkstra between waypoints...")
    ruta_total = [waypoints[0]]
    n_wp       = len(waypoints)

    for i in range(n_wp - 1):
        wp_a = waypoints[i]
        wp_b = waypoints[i+1]
        if wp_a == wp_b:
            continue

        idx_das = min(int(i * len(mascaras_validas) / (n_wp-1)),
                      len(mascaras_validas)-1)
        _, ruta_mask = mascaras_validas[idx_das]
        mask = cv2.imread(str(ruta_mask), cv2.IMREAD_GRAYSCALE)
        _, binary_tramo = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if binary_tramo.shape != (H_ref, W_ref):
            binary_tramo = cv2.resize(binary_tramo, (W_ref, H_ref),
                                      interpolation=cv2.INTER_NEAREST)
            _, binary_tramo = cv2.threshold(binary_tramo, 127, 255,
                                            cv2.THRESH_BINARY)

        dist_map_tramo = distance_transform_edt(binary_tramo)
        H_s       = H_ref // PASO
        W_s       = W_ref // PASO
        dist_small = cv2.resize(dist_map_tramo, (W_s, H_s),
                                interpolation=cv2.INTER_LINEAR)

        start_s = (min(wp_a[0]//PASO, H_s-1), min(wp_a[1]//PASO, W_s-1))
        end_s   = (min(wp_b[0]//PASO, H_s-1), min(wp_b[1]//PASO, W_s-1))

        tramo_small = dijkstra_dt(dist_small, start_s, end_s, H_s, W_s)
        tramo_full  = [(r*PASO, c*PASO) for r, c in tramo_small]
        ruta_total  = ruta_total[:-1] + tramo_full

    print(f"  Combined path: {len(ruta_total)} points")

    # --- FASE 3: Eliminar retrocesos ---
    ruta_limpia = eliminar_retrocesos(ruta_total, semilla_px, punta_px)
    print(f"  Path without reversals: {len(ruta_limpia)} points")

    # --- FASE 4: RDP ---
    ruta_rdp = rdp(ruta_limpia, RDP_EPSILON)
    print(f"  Path after RDP: {len(ruta_rdp)} points")

    # --- FASE 5: Suavizado ---
    ruta_final = suavizar_ruta(ruta_rdp, SUAVIZADO_VENTANA)

    # --- GUARDAR PKL ---
    out_linea = OUT_DIR / f"{prefijo}_central_line.pkl"
    with open(out_linea, "wb") as f:
        pickle.dump(ruta_final, f)

    # --- VISUALIZACION con escala automatica ---
    _, binary_final = cv2.threshold(mask_ref, 127, 255, cv2.THRESH_BINARY)
    scale = TARGET_HEIGHT_VIS / H_ref
    H_v   = TARGET_HEIGHT_VIS
    W_v   = int(W_ref * scale)

    vis   = np.zeros((H_v, W_v, 3), dtype=np.uint8)
    bin_v = cv2.resize(binary_final, (W_v, H_v),
                       interpolation=cv2.INTER_NEAREST)
    vis[bin_v > 0] = (60, 60, 60)

    for wp in waypoints:
        r_v, c_v = int(wp[0]*scale), int(wp[1]*scale)
        if 0 <= r_v < H_v and 0 <= c_v < W_v:
            cv2.circle(vis, (c_v, r_v), 4, (0, 255, 255), -1)

    for r, c in ruta_final:
        r_v, c_v = int(r*scale), int(c*scale)
        if 0 <= r_v < H_v and 0 <= c_v < W_v:
            cv2.circle(vis, (c_v, r_v), 2, (0, 0, 255), -1)

    cv2.circle(vis,
               (int(semilla_px[1]*scale), int(semilla_px[0]*scale)),
               7, (0,255,0), -1)
    cv2.circle(vis,
               (int(punta_px[1]*scale), int(punta_px[0]*scale)),
               7, (255,0,0), -1)

    cv2.imwrite(str(OUT_DIR / f"{prefijo}_vis.png"), vis)
    print(f"  Saved: {out_linea.name} | {prefijo}_vis.png")

print(f"\nProcessing complete. Results in: {OUT_DIR}")