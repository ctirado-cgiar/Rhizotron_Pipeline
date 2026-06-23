# 11_extraccionRasgos.py
import os
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
import pickle

# --- CONFIG ---
load_dotenv()

BASE          = os.path.dirname(os.getenv("CARPETA"))
CARPETA_MASK  = os.path.join(BASE, "08_rootPersistence")
CARPETA_GRAFO = os.path.join(BASE, "09_ragTemporal")
CARPETA_PIVOT = os.path.join(BASE, "10_mainRootIdentification")
OUT_DIR       = Path(os.path.join(BASE, "11_globalTraitsExtraction"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANCHO_RIZOTRON_CM = int(os.getenv("ANCHO_RIZOTRON_CM", 40))
ALTO_RIZOTRON_CM  = int(os.getenv("ALTO_RIZOTRON_CM", 60))
N_ZONAS           = int(os.getenv("N_ZONAS", 6))
DAS_MAX           = int(os.getenv("DAS_MAX", 26))

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
    print(f"Mask not found in: {CARPETA_MASK}")
    exit()
print(f"Rhizotrons detected ({len(rizotrones)}): {', '.join(rizotrones)}")

# --- FUNCIONES ---
def px_to_cm(px, escala):    return px / escala
def px2_to_cm2(px2, escala): return px2 / (escala ** 2)

def longitud_euclidiana(puntos):
    """Longitud real de una polilínea midiendo distancia euclidiana entre puntos consecutivos."""
    if len(puntos) < 2:
        return 0.0
    arr = np.array(puntos, dtype=float)
    diffs = np.diff(arr, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))

def longitud_esqueleto_euclidiana(skel):
    """
    Longitud real del esqueleto midiendo distancia euclidiana entre
    pixeles conectados. Mas preciso que contar pixeles.
    """
    ys, xs = np.where(skel > 0)
    if len(ys) < 2:
        return 0.0
    total = 0.0
    H, W = skel.shape
    vecinos_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    visitado = np.zeros((H, W), dtype=bool)
    for r, c in zip(ys, xs):
        if visitado[r, c]:
            continue
        for dr, dc in vecinos_8:
            nr, nc = r+dr, c+dc
            if (0 <= nr < H and 0 <= nc < W and skel[nr, nc] > 0
                    and not visitado[nr, nc]):
                total += np.sqrt(dr**2 + dc**2)
        visitado[r, c] = True
    return total

def tortuosidad(puntos):
    if len(puntos) < 2:
        return 1.0
    long_real  = longitud_euclidiana(puntos)
    dist_recta = np.hypot(puntos[-1][0]-puntos[0][0], puntos[-1][1]-puntos[0][1])
    return long_real / dist_recta if dist_recta > 1e-6 else 1.0

def contar_nodos_ramificacion(skel):
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    conv   = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
    return int(np.sum((conv >= 13) & (skel > 0)))

def diametro_pivotante(mask_pivotante, dist_map):
    ys, xs = np.where(mask_pivotante > 0)
    if len(ys) == 0:
        return 0.0, 0.0
    vals = dist_map[ys, xs] * 2
    vals = vals[vals > 0]
    return (float(np.max(vals)), float(np.min(vals))) if len(vals) else (0.0, 0.0)

def area_hull(binary):
    contornos, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return 0.0
    hull = cv2.convexHull(np.vstack(contornos))
    return cv2.contourArea(hull)

def ancho_maximo(binary):
    cols = np.where(np.any(binary > 0, axis=0))[0]
    return int(cols[-1] - cols[0]) if len(cols) else 0

def recortar_linea_a_profundidad(linea_arr, y_bot_px):
    """Recorta la linea central hasta el punto mas cercano a y_bot_px."""
    if len(linea_arr) == 0:
        return linea_arr
    ys = linea_arr[:, 0]
    # Encontrar hasta donde la linea llega en ese DAS
    idx = np.searchsorted(ys, y_bot_px)
    idx = min(idx, len(linea_arr) - 1)
    return linea_arr[:max(idx, 1)]

# --- PROCESAR CADA RIZOTRON ---
resultados_temporales = []
resultados_desempeno  = []

for prefijo in rizotrones:
    print(f"\n=== Rhizotron {prefijo} ===")

    partes   = prefijo.split("_", 1)
    plot     = partes[0]
    genotipo = partes[1] if len(partes) > 1 else prefijo

    grafo_path = Path(CARPETA_GRAFO) / f"{prefijo}_grafo.pkl"
    linea_path = Path(CARPETA_PIVOT) / f"{prefijo}__central_line.pkl"

    if not grafo_path.exists():
        print(f"  WARNING: lack of grafo para {prefijo}, will be skipped")
        continue
    if not linea_path.exists():
        print(f"  WARNING: lack of central line for {prefijo}, will be skipped")
        continue

    with open(grafo_path, "rb") as f:
        G = pickle.load(f)
    with open(linea_path, "rb") as f:
        linea_central = pickle.load(f)
    linea_arr = np.array(linea_central)

    # Reconstruir cadena pivotante por DAS
    das_unicos = sorted(set(d["das"] for _, d in G.nodes(data=True)
                            if d["das"] <= DAS_MAX))
    comps_primer = [(n, d) for n, d in G.nodes(data=True)
                    if d["das"] == das_unicos[0] and d["area_px"] >= 100]
    if not comps_primer:
        print(f"  WARNING: no initial components found for {prefijo}, will be skipped")
        continue

    origen_id = min(comps_primer, key=lambda x: x[1]["cy"])[0]
    das_a_nodo = {das_unicos[0]: origen_id}
    nodo_actual = origen_id
    cx_ref = G.nodes[origen_id]["cx"]

    for i in range(len(das_unicos) - 1):
        das_sig = das_unicos[i + 1]
        cand = [(n, d) for n, d in G.nodes(data=True)
                if d["das"] == das_sig
                and d["cy"] >= G.nodes[nodo_actual]["cy"] - 0.02
                and abs(d["cx"] - cx_ref) <= 0.20]
        if not cand:
            continue
        mejor_n, mejor_d = max(cand, key=lambda x: x[1]["cy"])
        das_a_nodo[das_sig] = mejor_n
        cx_ref = mejor_d["cx"]
        nodo_actual = mejor_n

    mascaras = [(d, r) for d, r in grupos[prefijo] if d <= DAS_MAX]
    if not mascaras:
        print(f"  WARNING: no masks valid")
        continue

    mask_ref  = cv2.imread(str(mascaras[-1][1]), cv2.IMREAD_GRAYSCALE)
    H_ref, W_ref = mask_ref.shape
    ESCALA_PX_CM = W_ref / ANCHO_RIZOTRON_CM

    zona_alto_cm = ALTO_RIZOTRON_CM / N_ZONAS
    zonas_rangos = [(i*zona_alto_cm, (i+1)*zona_alto_cm) for i in range(N_ZONAS)]
    zonas_px     = [(int(r0*ESCALA_PX_CM), int(r1*ESCALA_PX_CM))
                    for r0, r1 in zonas_rangos]

    prof_max_anterior = 0.0  # para monotonía
    filas_rizotron    = []

    for das, ruta_mask in mascaras:
        mask = cv2.imread(str(ruta_mask), cv2.IMREAD_GRAYSCALE)
        H, W = mask.shape
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if H != H_ref or W != W_ref:
            binary = cv2.resize(binary, (W_ref, H_ref), interpolation=cv2.INTER_NEAREST)
            _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

        area_px  = np.sum(binary > 0)
        area_cm2 = px2_to_cm2(area_px, ESCALA_PX_CM)
        hull_cm2 = px2_to_cm2(area_hull(binary), ESCALA_PX_CM)

        # --- PROFUNDIDAD desde grafo (monotónica) ---
        nodo_das = das_a_nodo.get(das)
        if nodo_das is not None:
            y_bot_norm = G.nodes[nodo_das].get("y_bot", G.nodes[nodo_das]["cy"])
            profundidad_px = y_bot_norm * H_ref
        else:
            profundidad_px = prof_max_anterior * ESCALA_PX_CM
        profundidad_cm = px_to_cm(profundidad_px, ESCALA_PX_CM)
        # Monotonía: nunca decrece
        profundidad_cm = max(profundidad_cm, prof_max_anterior)
        prof_max_anterior = profundidad_cm

        # --- LONGITUD PIVOTANTE desde linea central recortada ---
        if len(linea_arr) > 1:
            linea_das = recortar_linea_a_profundidad(linea_arr, profundidad_px)
            long_piv_cm  = px_to_cm(longitud_euclidiana(linea_das), ESCALA_PX_CM)
            tort_piv     = tortuosidad(linea_das)
        else:
            long_piv_cm = 0.0
            tort_piv    = 1.0

        # --- ESQUELETO COMPLETO (longitud euclidiana real) ---
        skel = skeletonize(binary // 255).astype(np.uint8)
        long_total_cm = px_to_cm(longitud_esqueleto_euclidiana(skel), ESCALA_PX_CM)
        n_nodos = contar_nodos_ramificacion(skel)

        # --- PIVOTANTE: dilatación de la línea central como máscara ---
        mask_piv_das = np.zeros((H_ref, W_ref), dtype=np.uint8)
        if len(linea_arr) > 1:
            linea_das_pts = recortar_linea_a_profundidad(linea_arr, profundidad_px)
            for r, c in linea_das_pts:
                if 0 <= int(r) < H_ref and 0 <= int(c) < W_ref:
                    cv2.circle(mask_piv_das, (int(c), int(r)), 3, 255, -1)

        skel_255 = skel * 255
        skel_lat = cv2.bitwise_and(
            skel_255,
            cv2.bitwise_not(cv2.dilate(mask_piv_das, np.ones((7,7), np.uint8))))
        long_lat_cm = px_to_cm(longitud_esqueleto_euclidiana(skel_lat > 0), ESCALA_PX_CM)

        num_labels_lat, _, stats_lat, _ = cv2.connectedComponentsWithStats(
            skel_lat, connectivity=8)
        n_laterales = sum(1 for i in range(1, num_labels_lat)
                          if stats_lat[i, cv2.CC_STAT_AREA] >= 10)
        densidad_ram = (n_laterales / long_piv_cm
                        if long_piv_cm > 0 else 0.0)

        ancho_cm  = px_to_cm(ancho_maximo(binary), ESCALA_PX_CM)
        dist_map  = distance_transform_edt(binary)
        dmax, dmin = diametro_pivotante(mask_piv_das, dist_map)
        diam_max_cm = px_to_cm(dmax, ESCALA_PX_CM)
        diam_min_cm = px_to_cm(dmin, ESCALA_PX_CM)

        fila = {
            "plot": plot, "genotipo": genotipo, "DAS": das,
            "area_total_cm2":         round(area_cm2, 3),
            "area_hull_cm2":          round(hull_cm2, 3),
            "profundidad_max_cm":     round(profundidad_cm, 2),
            "longitud_pivotante_cm":  round(long_piv_cm, 2),
            "tortuosidad_pivotante":  round(tort_piv, 3),
            "longitud_total_cm":      round(long_total_cm, 2),
            "longitud_laterales_cm":  round(long_lat_cm, 2),
            "n_laterales":            n_laterales,
            "densidad_ramificacion":  round(densidad_ram, 3),
            "n_nodos_ramificacion":   n_nodos,
            "ancho_max_cm":           round(ancho_cm, 2),
            "diametro_max_pivotante_cm": round(diam_max_cm, 3),
            "diametro_min_pivotante_cm": round(diam_min_cm, 3),
        }

        for idx, (r0_px, r1_px) in enumerate(zonas_px):
            r0_cm, r1_cm = zonas_rangos[idx]
            zona_mask    = binary[r0_px:r1_px, :]
            area_zona_px = np.sum(zona_mask > 0)
            nombre = f"{int(r0_cm)}_{int(r1_cm)}cm"
            fila[f"cobertura_{nombre}_cm2"] = round(
                px2_to_cm2(area_zona_px, ESCALA_PX_CM), 3)
            fila[f"cobertura_{nombre}_pct"] = round(
                area_zona_px / area_px * 100 if area_px > 0 else 0.0, 2)

        filas_rizotron.append(fila)

    resultados_temporales.extend(filas_rizotron)
    print(f"  Processing: {len(filas_rizotron)} DAS")

    # --- DESEMPEÑO ---
    if not filas_rizotron:
        continue

    df_riz  = pd.DataFrame(filas_rizotron)
    ultima  = df_riz.iloc[-1]

    # Tasas diarias (diferencia entre DAS consecutivos / días transcurridos)
    das_vals  = df_riz["DAS"].values
    prof_vals = df_riz["profundidad_max_cm"].values
    area_vals = df_riz["area_total_cm2"].values
    long_vals = df_riz["longitud_total_cm"].values

    tasas_prof, tasas_area, tasas_long = [], [], []
    for i in range(1, len(das_vals)):
        dt = das_vals[i] - das_vals[i-1]
        if dt > 0:
            tasas_prof.append((prof_vals[i] - prof_vals[i-1]) / dt)
            tasas_area.append((area_vals[i] - area_vals[i-1]) / dt)
            tasas_long.append((long_vals[i] - long_vals[i-1]) / dt)

    def safe_mean(lst): return round(float(np.mean(lst)), 3) if lst else 0.0
    def safe_std(lst):  return round(float(np.std(lst)),  3) if lst else 0.0

    resultados_desempeno.append({
        "plot": plot, "genotipo": genotipo,
        "area_final_cm2":                    ultima["area_total_cm2"],
        "tasa_crecimiento_area_promedio":     safe_mean(tasas_area),
        "tasa_crecimiento_area_std":          safe_std(tasas_area),
        "profundidad_final_cm":               ultima["profundidad_max_cm"],
        "tasa_profundizacion_promedio_cm_dia": safe_mean(tasas_prof),
        "tasa_profundizacion_std_cm_dia":     safe_std(tasas_prof),
        "longitud_total_final_cm":            ultima["longitud_total_cm"],
        "tasa_longitud_promedio_cm_dia":      safe_mean(tasas_long),
        "tasa_longitud_std_cm_dia":           safe_std(tasas_long),
        "longitud_pivotante_final_cm":        ultima["longitud_pivotante_cm"],
        "tortuosidad_pivotante_promedio":     round(df_riz["tortuosidad_pivotante"].mean(), 3),
        "ancho_max_final_cm":                 ultima["ancho_max_cm"],
        "n_laterales_final":                  ultima["n_laterales"],
        "densidad_ramificacion_promedio":     round(df_riz["densidad_ramificacion"].mean(), 3),
        "diametro_max_pivotante_cm":          round(df_riz["diametro_max_pivotante_cm"].max(), 3),
        "diametro_min_pivotante_cm":          round(df_riz["diametro_min_pivotante_cm"].min(), 3),
    })

# --- GUARDAR ---
df_temporal = pd.DataFrame(resultados_temporales)
df_temporal.to_csv(OUT_DIR / "rasgos_temporales.csv", index=False)
print(f"\nGuardado: rasgos_temporales.csv ({len(df_temporal)} filas)")

df_desempeno = pd.DataFrame(resultados_desempeno)
df_desempeno.to_csv(OUT_DIR / "rasgos_desempeno.csv", index=False)
print(f"Guardado: rasgos_desempeno.csv ({len(df_desempeno)} filas)")
print("\nListo.")