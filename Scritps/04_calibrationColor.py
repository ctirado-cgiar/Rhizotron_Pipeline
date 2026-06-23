# 04_calibrationColor.py
import os, cv2, warnings
import numpy as np
from plantcv import plantcv as pcv
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")
pcv.params.debug   = None
pcv.params.verbose = False

BASE   = os.path.dirname(os.getenv("CARPETA"))
SRC    = os.path.join(BASE, "03_dailySelection")
DST    = os.path.join(BASE, "04_calibrationColor")
ESCALA = 0.1
POS    = int(os.getenv("COLOR_POS", 3))  # probar 0,1,2,3
os.makedirs(DST, exist_ok=True)

for f in sorted(os.listdir(SRC)):
    if not f.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff')):
        continue

    path = os.path.join(SRC, f)
    try:
        img, _, _     = pcv.readimage(filename=path)
        H, W          = img.shape[:2]
        img_small     = cv2.resize(img, (int(W*ESCALA), int(H*ESCALA)))
        mascara_small = pcv.transform.detect_color_card(rgb_img=img_small)
        mascara_full  = cv2.resize(mascara_small, (W, H), interpolation=cv2.INTER_NEAREST)

        headers, card_matrix = pcv.transform.get_color_matrix(rgb_img=img, mask=mascara_full)
        std_matrix           = pcv.transform.std_color_matrix(pos=POS)

        img_corregida = pcv.transform.affine_color_correction(
            rgb_img       = img,
            source_matrix = card_matrix,
            target_matrix = std_matrix
        )

        img_bgr = cv2.cvtColor(img_corregida, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(DST, f), img_bgr)
        print(f"OK: {f}")
    except Exception as e:
        print(f"ERROR: {f} -> {e}")

print("¡Ready!")