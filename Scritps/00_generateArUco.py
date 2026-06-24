# 00_generateArUco.py
"""
ArUco marker generator for rhizotron perspective correction.
Generates 4 markers (IDs 0-3) for the 4 corners of the rhizotron frame.

Marker placement on rhizotron:
    ID0 (Top-Left)     |  ID1 (Top-Right)
    -------------------|-------------------
    ID3 (Bottom-Left)  |  ID2 (Bottom-Right)

NOTE: Marker total size INCLUDES the white border.
      Adjust MARKER_TOTAL_SIZE_CM to match your desired printed size.
"""

import cv2
import numpy as np
from cv2 import aruco
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ─── USER CONFIG ──────────────────────────────────────────────────────────────
# Output folder — change here or set ARUCO_OUT in your .env file
# Default: subfolder 'aruco_markers' next to this script
SCRIPT_DIR         = Path(__file__).parent
OUT_FOLDER         = Path(os.getenv("ARUCO_OUT", SCRIPT_DIR / "aruco_markers"))

# Marker size — total printed size INCLUDING white border (in cm)
MARKER_TOTAL_SIZE_CM = float(os.getenv("ARUCO_MARKER_SIZE_CM", 3.0))   # <-- change this to your desired size

# Border as fraction of total size (0.15 = 15% border on each side)
BORDER_FRACTION      = 0.15

# Print resolution
DPI = 300
# ──────────────────────────────────────────────────────────────────────────────

OUT_FOLDER.mkdir(parents=True, exist_ok=True)
CM_TO_INCH = 0.393701

# Derived sizes
total_px  = int(MARKER_TOTAL_SIZE_CM * CM_TO_INCH * DPI)
border_px = int(total_px * BORDER_FRACTION)
marker_px = total_px - 2 * border_px
marker_cm = MARKER_TOTAL_SIZE_CM * (1 - 2 * BORDER_FRACTION)

print("=" * 60)
print("ArUco Marker Generator for Rhizotron")
print("=" * 60)
print(f"Total marker size (incl. border): {MARKER_TOTAL_SIZE_CM} cm ({total_px} px)")
print(f"Inner code size:                  {marker_cm:.2f} cm ({marker_px} px)")
print(f"White border:                     {BORDER_FRACTION*100:.0f}% each side")
print(f"Resolution:                       {DPI} DPI")
print(f"Output folder:                    {OUT_FOLDER}")
print("=" * 60)

# ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Marker definitions: (id, corner_name, print_position_row, print_position_col)
# Placement order matches physical rhizotron corners
markers_info = [
    (0, "Top-Left",     0, 0),
    (1, "Top-Right",    0, 1),
    (2, "Bottom-Right", 1, 1),
    (3, "Bottom-Left",  1, 0),
]

# ─── GENERATE INDIVIDUAL MARKERS ──────────────────────────────────────────────
marker_images = {}

for marker_id, corner_name, _, _ in markers_info:
    # Generate marker
    code_img = aruco.generateImageMarker(aruco_dict, marker_id, marker_px)

    # Add white border
    img = np.ones((total_px, total_px), dtype=np.uint8) * 255
    img[border_px:border_px+marker_px,
        border_px:border_px+marker_px] = code_img

    marker_images[marker_id] = img

    # Save individual marker
    fname = OUT_FOLDER / f"aruco_ID{marker_id}_{corner_name.replace(' ','-')}_{MARKER_TOTAL_SIZE_CM}cm.png"
    cv2.imwrite(str(fname), img)
    print(f"  Saved: {fname.name}")

# ─── GENERATE PRINT GUIDE ─────────────────────────────────────────────────────
# A4 @ 300 DPI
a4_w = int(21   * CM_TO_INCH * DPI)
a4_h = int(29.7 * CM_TO_INCH * DPI)

sheet = np.ones((a4_h, a4_w), dtype=np.uint8) * 255

# Margins and spacing
margin_px  = int(2.0 * CM_TO_INCH * DPI)
gap_px     = int(1.5 * CM_TO_INCH * DPI)

# Calculate positions for 2x2 grid matching rhizotron layout
positions = {}
for marker_id, corner_name, row, col in markers_info:
    x = margin_px + col * (total_px + gap_px)
    y = margin_px + row * (total_px + gap_px)
    positions[marker_id] = (x, y)

# Place markers on sheet
font       = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
thickness  = 3
text_color = 0

for marker_id, corner_name, _, _ in markers_info:
    x, y = positions[marker_id]
    img  = marker_images[marker_id]

    # Paste marker
    sheet[y:y+total_px, x:x+total_px] = img

    # Label above marker
    label = f"ID {marker_id} - {corner_name}"
    cv2.putText(sheet, label,
                (x, y - int(0.3 * CM_TO_INCH * DPI)),
                font, font_scale, text_color, thickness)

    # Size note below marker
    size_note = f"{MARKER_TOTAL_SIZE_CM} x {MARKER_TOTAL_SIZE_CM} cm (total incl. border)"
    cv2.putText(sheet, size_note,
                (x, y + total_px + int(0.5 * CM_TO_INCH * DPI)),
                font, 0.8, text_color, 2)

# Draw rhizotron layout diagram
diagram_y = margin_px + 2 * (total_px + gap_px) + int(1.5 * CM_TO_INCH * DPI)
cv2.putText(sheet, "Rhizotron placement diagram:",
            (margin_px, diagram_y),
            font, font_scale, text_color, thickness)

diagram_y += int(1.0 * CM_TO_INCH * DPI)
diagram_lines = [
    " ___________________________",
    "|  ID0          |  ID1     |",
    "| (Top-Left)    |(Top-Right)|",
    "|               |          |",
    "|_______________|__________|",
    "|  ID3          |  ID2     |",
    "|(Bottom-Left)  |(Bot-Right)|",
    "|_______________|__________|",
]
for line in diagram_lines:
    cv2.putText(sheet, line,
                (margin_px, diagram_y),
                cv2.FONT_HERSHEY_PLAIN, 2.0, text_color, 2)
    diagram_y += int(0.7 * CM_TO_INCH * DPI)

# Printing instructions
instr_y = diagram_y + int(1.0 * CM_TO_INCH * DPI)
instructions = [
    "PRINTING INSTRUCTIONS:",
    f"1. Print at {DPI} DPI on A4 paper",
    f"2. Each marker total size: {MARKER_TOTAL_SIZE_CM} x {MARKER_TOTAL_SIZE_CM} cm (includes white border)",
    f"3. Inner code size: {marker_cm:.2f} x {marker_cm:.2f} cm",
    "4. Laminate or use waterproof paper",
    "5. Place on rhizotron frame corners as shown in diagram above",
    "6. Ensure markers are flat, unobstructed and well-lit during imaging",
]
for instruction in instructions:
    cv2.putText(sheet, instruction,
                (margin_px, instr_y),
                font, 0.9, text_color, 2)
    instr_y += int(0.8 * CM_TO_INCH * DPI)

# Save print guide
guide_path = OUT_FOLDER / f"aruco_print_guide_{MARKER_TOTAL_SIZE_CM}cm.png"
cv2.imwrite(str(guide_path), sheet)
print(f"  Saved: {guide_path.name}")

print("=" * 60)
print("DONE. Files generated:")
for marker_id, corner_name, _, _ in markers_info:
    print(f"  ID{marker_id} ({corner_name})")
print(f"  Print guide: {guide_path.name}")
print(f"\nAll files saved to: {OUT_FOLDER}")
print("=" * 60)