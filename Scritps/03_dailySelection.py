# 03_dailySelection.py
import os, re, shutil, cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BASE = os.path.dirname(os.getenv("CARPETA"))
SRC  = os.path.join(BASE, "02_rename")
DST  = os.path.join(BASE, "03_dailySelection")
os.makedirs(DST, exist_ok=True)

def brillo(path):
    return np.mean(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

# Agrupar por rizotron y por dia
archivos = [f for f in os.listdir(SRC) if f.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff'))]

por_rizotron = {}
por_dia      = {}

for f in archivos:
    m = re.match(r'^(.+)_(DAS\d+)(_\d+)?\.\w+$', f)
    if not m: continue
    rizotron = m.group(1)
    clave    = f"{rizotron}_{m.group(2)}"
    por_rizotron.setdefault(rizotron, []).append(f)
    por_dia.setdefault(clave, []).append(f)

# Brillo histórico por rizotron
brillo_historico = {
    r: np.mean([brillo(os.path.join(SRC, f)) for f in files])
    for r, files in por_rizotron.items()
}

# Seleccionar la más cercana al brillo histórico y guardar con nombre limpio
for clave, archivos_dia in sorted(por_dia.items()):
    rizotron = re.match(r'^(.+)_DAS\d+$', clave).group(1)
    objetivo = brillo_historico[rizotron]
    brillos  = np.array([brillo(os.path.join(SRC, f)) for f in archivos_dia])
    elegida  = archivos_dia[np.argmin(np.abs(brillos - objetivo))]
    ext      = os.path.splitext(elegida)[1]
    nombre   = clave + ext  # nombre limpio sin _N
    shutil.copy2(os.path.join(SRC, elegida), os.path.join(DST, nombre))
    print(f"OK: {elegida} -> {nombre}")

print(f"\n{len(por_dia)} images selected.")