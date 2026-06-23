# 02_rename.py
import os, shutil, datetime, re, cv2
from PIL import Image
from PIL.ExifTags import TAGS
from pyzbar.pyzbar import decode
from dotenv import load_dotenv

load_dotenv()

BASE          = os.path.dirname(os.getenv("CARPETA"))
SRC           = os.path.join(BASE, "01_homography")
DST           = os.path.join(BASE, "02_rename")
FECHA_SIEMBRA = datetime.datetime.strptime(os.getenv("FECHA_SIEMBRA"), "%Y-%m-%d")
os.makedirs(DST, exist_ok=True)

def fecha_exif(path):
    try:
        exif = Image.open(path)._getexif()
        if exif:
            for tid, val in exif.items():
                if TAGS.get(tid) == "DateTimeOriginal":
                    return datetime.datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
    except: pass
    return None

def fecha_nombre(f):
    m = re.search(r'(\d{8})_(\d{6})', f)
    if m:
        try: return datetime.datetime.strptime(m.group(1)+m.group(2), "%Y%m%d%H%M%S")
        except: pass
    return None

def get_fecha(path, f):
    return fecha_exif(path) or fecha_nombre(f) or datetime.datetime.fromtimestamp(os.path.getctime(path))

def leer_qr(path):
    try:
        img  = cv2.imread(path)
        H, W = img.shape[:2]

        # Recortar zona inferior central donde siempre está el QR
        zona = img[int(H*0.75):H, int(W*0.25):int(W*0.75)]

        # Intentar varias escalas
        for escala in [2, 3, 1]:
            h, w     = zona.shape[:2]
            ampliada = cv2.resize(zona, (w*escala, h*escala))
            codigos  = decode(Image.fromarray(cv2.cvtColor(ampliada, cv2.COLOR_BGR2RGB)))
            for c in codigos:
                raw  = c.data.decode("utf-8").strip()
                part = re.split(r'\s+|\d{2}/\d{2}/\d{4}', raw)[0].strip()
                part = re.sub(r'[\\/:*?"<>|]', '-', part)
                part = part.strip('-_ ')
                part = re.sub(r'_+', '_', part)
                if part: return part
    except: pass
    return None

for f in sorted(os.listdir(SRC)):
    if not f.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff')):
        continue

    path  = os.path.join(SRC, f)
    fecha = get_fecha(path, f)
    qr    = leer_qr(path)
    das   = (fecha.date() - FECHA_SIEMBRA.date()).days
    ext   = os.path.splitext(f)[1]

    prefijo = qr if qr else "NOQR"
    nombre  = f"{prefijo}_DAS{das}{ext}"

    base, i = os.path.splitext(nombre)[0], 1
    while os.path.exists(os.path.join(DST, nombre)):
        nombre, i = f"{base}_{i}{ext}", i+1

    shutil.copy2(path, os.path.join(DST, nombre))
    print(f"{f} -> 02_rename/{nombre}")

print("¡Ready!")