# 11_globalTraitsExtraction.py
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

ANCHO_RIZOTRON_CM  = float(os.getenv("ANCHO_RIZOTRON_CM", 40))
ALTO_RIZOTRON_CM   = float(os.getenv("ALTO_RIZOTRON_CM", 60))
N_ZONAS            = int(os.getenv("N_ZONAS", 6))
DAS_MAX            = int(os.getenv("DAS_MAX", 26))
MIN_LARGO_LATERAL  = 15   # minimum pixels to consider a lateral root

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
    print(f"No masks found in: {CARPETA_MASK}")
    exit()
print(f"Rhizotrons detected ({len(rizotrones)}): {', '.join(rizotrones)}")

# --- FUNCIONES GENERALES ---
def px_to_cm(px, escala):    return px / escala
def px2_to_cm2(px2, escala): return px2 / (escala ** 2)

def longitud_euclidiana(puntos):
    if len(puntos) < 2:
        return 0.0
    arr   = np.array(puntos, dtype=float)
    diffs = np.diff(arr, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))

def longitud_esqueleto_euclidiana(skel):
    ys, xs = np.where(skel > 0)
    if len(ys) < 2:
        return 0.0
    total    = 0.0
    H, W     = skel.shape
    v8       = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    visitado = np.zeros((H, W), dtype=bool)
    for r, c in zip(ys, xs):
        if visitado[r, c]:
            continue
        for dr, dc in v8:
            nr, nc = r+dr, c+dc
            if (0 <= nr < H and 0 <= nc < W
                    and skel[nr, nc] > 0 and not visitado[nr, nc]):
                total += np.sqrt(dr**2 + dc**2)
        visitado[r, c] = True
    return total

def tortuosidad(puntos):
    if len(puntos) < 2:
        return 1.0
    long_real  = longitud_euclidiana(puntos)
    dist_recta = np.hypot(puntos[-1][0]-puntos[0][0],
                          puntos[-1][1]-puntos[0][1])
    return long_real / dist_recta if dist_recta > 1e-6 else 1.0

def contar_nodos_ramificacion(skel):
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    conv   = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
    return int(np.sum((conv >= 13) & (skel > 0)))

def diametro_pivotante(mask_piv, dist_map):
    ys, xs = np.where(mask_piv > 0)
    if len(ys) == 0:
        return 0.0, 0.0
    vals = dist_map[ys, xs] * 2
    vals = vals[vals > 0]
    return (float(np.max(vals)), float(np.min(vals))) if len(vals) else (0.0, 0.0)

def area_hull(binary):
    contornos, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return 0.0
    hull = cv2.convexHull(np.vstack(contornos))
    return cv2.contourArea(hull)

def ancho_maximo(binary):
    cols = np.where(np.any(binary > 0, axis=0))[0]
    return int(cols[-1] - cols[0]) if len(cols) else 0

def recortar_linea_a_profundidad(linea_arr, y_bot_px):
    if len(linea_arr) == 0:
        return linea_arr
    ys  = linea_arr[:, 0]
    idx = np.searchsorted(ys, y_bot_px)
    idx = min(idx, len(linea_arr) - 1)
    return linea_arr[:max(idx, 1)]

# --- FUNCION: ANGULOS DE LATERALES ---
def extraer_angulos_laterales(skel_lat, mask_piv_das, linea_arr,
                               plot, genotipo, das,
                               H, W, ESCALA_PX_CM,
                               zona_alto_px, N_ZONAS,
                               ALTO_RIZOTRON_CM):
    """
    Para cada componente lateral del esqueleto:
    - Encuentra el punto mas cercano al pivotante (origen)
    - Encuentra el punto mas lejano (punta)
    - Calcula angulo respecto a la vertical, longitud, zona y orden estimado
    """
    registros = []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skel_lat, connectivity=8)

    # Umbral de orden: mediana de longitudes de todos los componentes
    longitudes_px = [stats[i, cv2.CC_STAT_AREA]
                     for i in range(1, num_labels)
                     if stats[i, cv2.CC_STAT_AREA] >= MIN_LARGO_LATERAL]
    if not longitudes_px:
        return registros
    umbral_orden = float(np.median(longitudes_px))

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < MIN_LARGO_LATERAL:
            continue

        pixeles = np.argwhere(labels == i)  # (r, c)

        # Origen: pixel mas cercano al pivotante
        if len(linea_arr) > 1:
            dist_al_piv = np.array([
                np.min(np.sqrt(
                    (linea_arr[:, 0] - p[0])**2 +
                    (linea_arr[:, 1] - p[1])**2
                )) for p in pixeles
            ])
            idx_origen = np.argmin(dist_al_piv)
        else:
            idx_origen = 0
        origen = pixeles[idx_origen]

        # Punta: pixel mas lejano del origen
        dists_desde_origen = np.sqrt(
            (pixeles[:, 0] - origen[0])**2 +
            (pixeles[:, 1] - origen[1])**2
        )
        idx_punta    = np.argmax(dists_desde_origen)
        punta        = pixeles[idx_punta]
        longitud_px  = dists_desde_origen[idx_punta]

        if longitud_px < MIN_LARGO_LATERAL:
            continue

        longitud_cm = px_to_cm(longitud_px, ESCALA_PX_CM)

        # Angulo respecto a la vertical (0=vertical, 90=horizontal)
        dy = punta[0] - origen[0]
        dx = punta[1] - origen[1]
        angulo_deg = float(np.degrees(np.arctan2(abs(dx), abs(dy))))
        direccion  = "right" if dx >= 0 else "left"

        # Zona de profundidad segun origen
        zona_idx  = min(int(origen[0] / zona_alto_px), N_ZONAS - 1)
        zona_cm0  = int(zona_idx * ALTO_RIZOTRON_CM / N_ZONAS)
        zona_cm1  = int((zona_idx + 1) * ALTO_RIZOTRON_CM / N_ZONAS)
        zona_name = f"{zona_cm0}_{zona_cm1}cm"

        # Orden estimado por longitud
        orden = "1" if longitud_px >= umbral_orden else "2"

        registros.append({
            "plot":        plot,
            "genotipo":    genotipo,
            "DAS":         das,
            "lateral_id":  i,
            "origen_r_px": int(origen[0]),
            "origen_c_px": int(origen[1]),
            "punta_r_px":  int(punta[0]),
            "punta_c_px":  int(punta[1]),
            "longitud_cm": round(longitud_cm, 3),
            "angulo_deg":  round(angulo_deg, 2),
            "direccion":   direccion,
            "zona":        zona_name,
            "orden":       orden,
        })

    return registros

# --- PROCESAR CADA RIZOTRON ---
resultados_temporales = []
resultados_desempeno  = []
resultados_angulos    = []

for prefijo in rizotrones:
    print(f"\n=== Rhizotron {prefijo} ===")

    partes   = prefijo.split("_", 1)
    plot     = partes[0]
    genotipo = partes[1] if len(partes) > 1 else prefijo

    grafo_path   = Path(CARPETA_GRAFO) / f"{prefijo}_grafo.pkl"
    piv_path     = Path(CARPETA_GRAFO) / f"{prefijo}_pivotant.pkl"
    linea_path   = Path(CARPETA_PIVOT) / f"{prefijo}_central_line.pkl"

    if not grafo_path.exists():
        print(f"  WARNING: graph not found for {prefijo}, skipping")
        continue
    if not piv_path.exists():
        print(f"  WARNING: taproot chain not found for {prefijo}, skipping")
        continue
    if not linea_path.exists():
        print(f"  WARNING: central line not found for {prefijo}, skipping")
        continue

    with open(grafo_path, "rb") as f:
        G = pickle.load(f)
    with open(piv_path, "rb") as f:
        camino_pivotante = pickle.load(f)
    with open(linea_path, "rb") as f:
        linea_central = pickle.load(f)

    linea_arr = np.array(linea_central)

    # Mapear DAS -> nodo del pivotante (desde pkl, sin recalcular)
    das_a_nodo = {G.nodes[n]["das"]: n for n in camino_pivotante}

    mascaras = [(d, r) for d, r in grupos[prefijo] if d <= DAS_MAX]
    if not mascaras:
        print(f"  WARNING: no valid masks found for {prefijo}, skipping")
        continue

    mask_ref     = cv2.imread(str(mascaras[-1][1]), cv2.IMREAD_GRAYSCALE)
    H_ref, W_ref = mask_ref.shape
    ESCALA_PX_CM = W_ref / ANCHO_RIZOTRON_CM

    zona_alto_cm = ALTO_RIZOTRON_CM / N_ZONAS
    zona_alto_px = int(H_ref / N_ZONAS)
    zonas_rangos = [(i*zona_alto_cm, (i+1)*zona_alto_cm) for i in range(N_ZONAS)]
    zonas_px     = [(int(r0*ESCALA_PX_CM), int(r1*ESCALA_PX_CM))
                    for r0, r1 in zonas_rangos]

    prof_max_anterior = 0.0
    filas_rizotron    = []

    # Angulos solo del DAS final (mascara mas completa)
    das_final = mascaras[-1][0]

    for das, ruta_mask in mascaras:
        mask = cv2.imread(str(ruta_mask), cv2.IMREAD_GRAYSCALE)
        H, W = mask.shape
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        if H != H_ref or W != W_ref:
            binary = cv2.resize(binary, (W_ref, H_ref),
                                interpolation=cv2.INTER_NEAREST)
            _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

        area_px  = np.sum(binary > 0)
        area_cm2 = px2_to_cm2(area_px, ESCALA_PX_CM)
        hull_cm2 = px2_to_cm2(area_hull(binary), ESCALA_PX_CM)

        # Profundidad desde grafo (monotonica)
        nodo_das = das_a_nodo.get(das)
        if nodo_das is not None:
            y_bot_norm     = G.nodes[nodo_das].get("y_bot",
                                                    G.nodes[nodo_das]["cy"])
            profundidad_px = y_bot_norm * H_ref
        else:
            profundidad_px = prof_max_anterior * ESCALA_PX_CM
        profundidad_cm    = px_to_cm(profundidad_px, ESCALA_PX_CM)
        profundidad_cm    = max(profundidad_cm, prof_max_anterior)
        prof_max_anterior = profundidad_cm

        # Longitud pivotante desde linea central recortada
        if len(linea_arr) > 1:
            linea_das    = recortar_linea_a_profundidad(linea_arr, profundidad_px)
            long_piv_cm  = px_to_cm(longitud_euclidiana(linea_das), ESCALA_PX_CM)
            tort_piv     = tortuosidad(linea_das)
        else:
            long_piv_cm = 0.0
            tort_piv    = 1.0

        # Esqueleto completo
        skel          = skeletonize(binary // 255).astype(np.uint8)
        long_total_cm = px_to_cm(longitud_esqueleto_euclidiana(skel), ESCALA_PX_CM)
        n_nodos       = contar_nodos_ramificacion(skel)

        # Mascara del pivotante
        mask_piv_das = np.zeros((H_ref, W_ref), dtype=np.uint8)
        if len(linea_arr) > 1:
            linea_das_pts = recortar_linea_a_profundidad(linea_arr, profundidad_px)
            for r, c in linea_das_pts:
                if 0 <= int(r) < H_ref and 0 <= int(c) < W_ref:
                    cv2.circle(mask_piv_das, (int(c), int(r)), 3, 255, -1)

        # Esqueleto de laterales
        skel_255 = skel * 255
        skel_lat = cv2.bitwise_and(
            skel_255,
            cv2.bitwise_not(cv2.dilate(mask_piv_das,
                                        np.ones((7,7), np.uint8))))

        long_lat_cm = px_to_cm(longitud_esqueleto_euclidiana(skel_lat > 0),
                                ESCALA_PX_CM)

        num_labels_lat, _, stats_lat, _ = cv2.connectedComponentsWithStats(
            skel_lat, connectivity=8)
        n_laterales  = sum(1 for i in range(1, num_labels_lat)
                           if stats_lat[i, cv2.CC_STAT_AREA] >= 10)
        densidad_ram = n_laterales / long_piv_cm if long_piv_cm > 0 else 0.0

        ancho_cm    = px_to_cm(ancho_maximo(binary), ESCALA_PX_CM)
        dist_map    = distance_transform_edt(binary)
        dmax, dmin  = diametro_pivotante(mask_piv_das, dist_map)
        diam_max_cm = px_to_cm(dmax, ESCALA_PX_CM)
        diam_min_cm = px_to_cm(dmin, ESCALA_PX_CM)

        fila = {
            "plot": plot, "genotipo": genotipo, "DAS": das,
            "area_total_cm2":            round(area_cm2, 3),
            "area_hull_cm2":             round(hull_cm2, 3),
            "profundidad_max_cm":        round(profundidad_cm, 2),
            "longitud_pivotante_cm":     round(long_piv_cm, 2),
            "tortuosidad_pivotante":     round(tort_piv, 3),
            "longitud_total_cm":         round(long_total_cm, 2),
            "longitud_laterales_cm":     round(long_lat_cm, 2),
            "n_laterales":               n_laterales,
            "densidad_ramificacion":     round(densidad_ram, 3),
            "n_nodos_ramificacion":      n_nodos,
            "ancho_max_cm":              round(ancho_cm, 2),
            "diametro_max_pivotante_cm": round(diam_max_cm, 3),
            "diametro_min_pivotante_cm": round(diam_min_cm, 3),
        }

        for idx, (r0_px, r1_px) in enumerate(zonas_px):
            r0_cm, r1_cm = zonas_rangos[idx]
            zona_mask    = binary[r0_px:r1_px, :]
            area_zona_px = np.sum(zona_mask > 0)
            nombre       = f"{int(r0_cm)}_{int(r1_cm)}cm"
            fila[f"cobertura_{nombre}_cm2"] = round(
                px2_to_cm2(area_zona_px, ESCALA_PX_CM), 3)
            fila[f"cobertura_{nombre}_pct"] = round(
                area_zona_px / area_px * 100 if area_px > 0 else 0.0, 2)

        filas_rizotron.append(fila)

        # Angulos de laterales solo en el DAS final
        if das == das_final:
            angulos = extraer_angulos_laterales(
                skel_lat, mask_piv_das, linea_arr,
                plot, genotipo, das,
                H_ref, W_ref, ESCALA_PX_CM,
                zona_alto_px, N_ZONAS, ALTO_RIZOTRON_CM
            )
            resultados_angulos.extend(angulos)
            print(f"  Lateral angles extracted: {len(angulos)} laterals")

    resultados_temporales.extend(filas_rizotron)
    print(f"  Temporal traits: {len(filas_rizotron)} DAS processed")

    # Desempeno
    if not filas_rizotron:
        continue

    df_riz  = pd.DataFrame(filas_rizotron)
    ultima  = df_riz.iloc[-1]

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
        "plot":                           plot,
        "genotipo":                       genotipo,
        "area_final_cm2":                 ultima["area_total_cm2"],
        "tasa_crecimiento_area_promedio": safe_mean(tasas_area),
        "tasa_crecimiento_area_std":      safe_std(tasas_area),
        "profundidad_final_cm":           ultima["profundidad_max_cm"],
        "tasa_profundizacion_promedio_cm_dia": safe_mean(tasas_prof),
        "tasa_profundizacion_std_cm_dia": safe_std(tasas_prof),
        "longitud_total_final_cm":        ultima["longitud_total_cm"],
        "tasa_longitud_promedio_cm_dia":  safe_mean(tasas_long),
        "tasa_longitud_std_cm_dia":       safe_std(tasas_long),
        "longitud_pivotante_final_cm":    ultima["longitud_pivotante_cm"],
        "tortuosidad_pivotante_promedio": round(
            df_riz["tortuosidad_pivotante"].mean(), 3),
        "ancho_max_final_cm":             ultima["ancho_max_cm"],
        "n_laterales_final":              ultima["n_laterales"],
        "densidad_ramificacion_promedio": round(
            df_riz["densidad_ramificacion"].mean(), 3),
        "diametro_max_pivotante_cm":      round(
            df_riz["diametro_max_pivotante_cm"].max(), 3),
        "diametro_min_pivotante_cm":      round(
            df_riz["diametro_min_pivotante_cm"].min(), 3),
    })

# --- GUARDAR ---
df_temporal = pd.DataFrame(resultados_temporales)
df_temporal.to_csv(OUT_DIR / "traits_temporals.csv", index=False)
print(f"\nSaved: traits_temporals.csv ({len(df_temporal)} rows)")

df_desempeno = pd.DataFrame(resultados_desempeno)
df_desempeno.to_csv(OUT_DIR / "traits_performance.csv", index=False)
print(f"Saved: traits_performance.csv ({len(df_desempeno)} rows)")

df_angulos = pd.DataFrame(resultados_angulos)
df_angulos.to_csv(OUT_DIR / "angles_laterals.csv", index=False)
print(f"Saved: angles_laterals.csv ({len(df_angulos)} rows)")

print("\nDone.")