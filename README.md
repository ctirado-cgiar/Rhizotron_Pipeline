# Rhizotron Pipeline

A computer vision pipeline for temporal root system architecture (RSA) phenotyping in rhizotron experiments. It processes time-series rhizotron images from raw capture to structured trait databases, enabling quantitative characterization of root growth dynamics across genotypes.

> **Example data:** The scripts, sample images, and reference data included in this repository correspond to a *Phaseolus spp.* experiment conducted at CIAT (Palmira, Colombia). The pipeline is designed to work with any crop species imaged in rhizotrons captured under standard conditions described here.

---

## What it does

- Corrects geometric distortion using ArUco markers
- Segments root structures using a SegFormer deep learning model (Roboflow)
- Reconstructs binary masks and accumulates them over time
- Identifies the taproot centerline using Dijkstra's algorithm
- Extracts 20+ root architecture traits per experimental unit and timepoint
- Outputs structured CSV databases ready for statistical analysis

---

## Pipeline overview

```
Raw images
    │
    ▼
01_homographyArUco.py       Geometric correction using ArUco markers
    │
    ▼
02_renameQrCode.py          QR-based automatic image renaming (plot + DAS)
    │
    ▼
03_dailySelection.py        Automatic selection of best daily image per rhizotron
    │
    ▼
04_calibrationColor.py      Colorimetric calibration using ColorChecker
    │
    ▼
05_patchesGeneration.py     Image tiling into 512x512 patches for segmentation
    │
    ▼
06a_cloudRootSeg.py         Root segmentation via Roboflow cloud API
06b_localRootSeg.py         Root segmentation via local inference
    │
    ▼
07_patchesReconstruccion.py Patch reconstruction into full binary masks
    │
    ▼
08_rootPersistence.py       Temporal OR accumulation of binary masks per DAS
    │
    ▼
09_ragTemporal.py           Temporal Region Adjacency Graph (RAG) + Hungarian
                            Algorithm for inter-timepoint component matching
                            and taproot chain identification
    │
    ▼
10a_addSeedTip.py           Interactive tool: mark seed position and taproot tip
                            per rhizotron (one-time manual step)
    │
    ▼
10b_mainRootIdentification.py  Taproot centerline reconstruction using Dijkstra
                               on distance transform + RDP simplification
    │
    ▼
11_globalTraitsExtraction.py   Extraction of 20+ RSA traits per unit and DAS
                               → traits_temporals.csv
                               → traits_performance.csv
                               → angles_laterals.csv
    │
    ▼                              
12_analisysGraphs.r       Statistical analysis and figure generation in R:
                          summary table and three figures (penetration
                          scatter, root coverage profile, root architecture
                          schemes) by genotype, generated dynamically from
                          → tabla3_performance_genotipo.png
                          → scatter_penetracion.png
                          → perfil_cobertura.png
                          → esquemas_panel.png + esquemas_individuales/
```

---

## Output databases

| File | Description |
|------|-------------|
| `traits_temporals.csv` | Root system traits per experimental unit and DAS |
| `traits_performance.csv` | Summary performance indicators per unit across the full trial |
| `angles_laterals.csv` | Individual lateral root measurements: insertion angle, length, depth zone |
| `12_analisysGraphs/` (R outputs) | Summary statistics table and figures by genotype, derived from the three CSV databases above |

---

## Requirements

- Python 3.10+
- See `requirements.txt` for all dependencies

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ctirado-cgiar/Rhizotron_Pipeline.git
cd Rhizotron_Pipeline

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/Mac
# Edit .env with your paths and parameters
```

---

## Configuration

All user-specific paths and parameters are defined in a `.env` file.
Copy `.env.example` and edit accordingly:

```env
CARPETA=D:/your/path/to/trial/00_originales
DAS_MAX=26
ARUCO_MARKER_SIZE_CM=3.0
ROBOFLOW_API_KEY=your_api_key_here
ROBOFLOW_MODEL_ID=rootseg-mmejg/5
```

---

## ArUco markers

Run `00_generateArUco.py` once to generate printable ArUco markers for your rhizotron frame. Marker size is configurable via `ARUCO_MARKER_SIZE_CM` in `.env`.

Marker placement on the rhizotron frame:

```
ID0 (Top-Left)    |  ID1 (Top-Right)
------------------|------------------
ID3 (Bottom-Left) |  ID2 (Bottom-Right)
```

---

## Manual steps

Two steps require human input and are run once per trial:

- **`00_generateArUco.py`** — generate and print ArUco markers before image acquisition
- **`10a_addSeedTip.py`** — interactively mark seed position and taproot tip on the final accumulated mask for each rhizotron

All other steps are fully automated.

---

## Citation 

If you use this pipeline in your research, please cite:

Citation information will be available once the manuscript and pipeline documentation are finalized. 
> Tirado-Murcia, C., Aragón, J.E., & Polania, J.A. (2026). *RhizoSight: A computer vision pipeline for temporal root system architecture phenotyping in rhizotrons*. Alliance of Bioversity International & CIAT, Palmira, Colombia.

---

## Authors

**Cristian Tirado-Murcia** - c.tirado@cgiar.org  

**Jorge Aragón Medina**    - j.aragon@cgiar.org

**José Polania**           - j.polania@cgiar.org

Alliance of Bioversity International & CIAT — Bean Program  
Palmira, Colombia

---

## Founding
This research was carried out within the framework of the Breakthrough project, supported by the **Gates Foundation.**

---
## License

This project is licensed under ...
