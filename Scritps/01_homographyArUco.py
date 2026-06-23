# 01_homographyArUco.py
import os, cv2, struct
import numpy as np
from dotenv import load_dotenv

load_dotenv()

SRC    = os.getenv("CARPETA")
DST    = os.path.join(os.path.dirname(SRC), "01_homography")
MARGEN = int(os.getenv("MARGEN", 10))
os.makedirs(DST, exist_ok=True)

DICT     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
corner_idx = {0: 2, 1: 3, 2: 0, 3: 1}

def get_esquinas(img):
    corners, ids, _ = DETECTOR.detectMarkers(img)
    if ids is None or len(ids) < 4:
        return None
    esq = {}
    for i, mid in enumerate(ids.flatten()):
        if mid in corner_idx:
            esq[mid] = corners[i][0][corner_idx[mid]]
    return esq if len(esq) == 4 else None

archivos = sorted([f for f in os.listdir(SRC) if f.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff'))])

for f in archivos:
    path = os.path.join(SRC, f)
    img  = cv2.imread(path)
    esq  = get_esquinas(img)

    if esq is None:
        print(f"NOARUCO (skipped): {f}")
        
        continue

    p0, p1, p2, p3 = esq[0], esq[1], esq[2], esq[3]

    # Calcular ancho y alto reales desde los ArUcos
    ancho = int((np.linalg.norm(p1-p0) + np.linalg.norm(p2-p3)) / 2)
    alto  = int((np.linalg.norm(p3-p0) + np.linalg.norm(p2-p1)) / 2)

    # Agregar margen
    W_out = ancho + MARGEN * 2
    H_out = alto  + MARGEN * 2

    src_pts = np.float32([p0, p1, p2, p3])
    dst_pts = np.float32([
        [MARGEN,        MARGEN       ],
        [W_out-MARGEN,  MARGEN       ],
        [W_out-MARGEN,  H_out-MARGEN ],
        [MARGEN,        H_out-MARGEN ],
    ])

    M      = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(img, M, (W_out, H_out))
    cv2.imwrite(os.path.join(DST, f), result)
    print(f"OK: {f} -> {W_out}x{H_out} px")

print("¡Ready!")