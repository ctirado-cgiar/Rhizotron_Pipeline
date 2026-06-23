# 05_patchesGeneration.py
import os, cv2
from dotenv import load_dotenv

load_dotenv()

BASE       = os.path.dirname(os.getenv("CARPETA"))
SRC        = os.path.join(BASE, "03_dailySelection")
DST        = os.path.join(BASE, "05_patchesGeneration")
PATCH_SIZE = int(os.getenv("PATCH_SIZE", 512))
OVERLAP    = int(os.getenv("OVERLAP", 32))
MIN_VAR    = float(os.getenv("MIN_VARIANZA", 50))
os.makedirs(DST, exist_ok=True)

step = PATCH_SIZE - OVERLAP

for f in sorted(os.listdir(SRC)):
    if not f.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff')):
        continue

    img  = cv2.imread(os.path.join(SRC, f))
    H, W = img.shape[:2]
    base = os.path.splitext(f)[0]
    ext  = os.path.splitext(f)[1]
    saved = 0

    # Posiciones regulares en Y, asegurando cubrir hasta el borde inferior
    ys = list(range(0, H - PATCH_SIZE + 1, step))
    if not ys or ys[-1] + PATCH_SIZE < H:
        ys.append(H - PATCH_SIZE)  # patch alineado al borde inferior

    # Posiciones regulares en X, asegurando cubrir hasta el borde derecho
    xs = list(range(0, W - PATCH_SIZE + 1, step))
    if not xs or xs[-1] + PATCH_SIZE < W:
        xs.append(W - PATCH_SIZE)  # patch alineado al borde derecho

    # Evitar coordenadas duplicadas (cuando el último patch ya coincide con el borde)
    ys = sorted(set(ys))
    xs = sorted(set(xs))

    for y in ys:
        for x in xs:
            patch    = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            varianza = cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if varianza < MIN_VAR:
                continue
            nombre = f"{base}__y{y}_x{x}{ext}"
            cv2.imwrite(os.path.join(DST, nombre), patch)
            saved += 1

    print(f"{f} -> {saved} patches")

print("¡Ready!")