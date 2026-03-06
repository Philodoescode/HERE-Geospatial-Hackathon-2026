# HERE Geospatial Hackathon — Road Centerline Extraction

> **3rd Place** at the HERE Geospatial Hackathon  
> **Precision: 99.8% | Recall: ~71.3%**

This repository contains our solution for automatically extracting road centerlines from GPS probe data (VPD/HPD) over Kosovo. The pipeline reconstructs a clean, topologically correct road network and evaluates it against HERE Nav Streets ground truth.

---

## Results

| Metric | Score |
|---|---|
| Precision | **99.8%** |
| Recall | **~71.3%** |
| Final Placement | **3rd Place** |

Evaluation is done using segmented precision/recall: both generated and reference lines are split into ~20 m chunks, and a match is counted when a chunk falls within a 15 m buffer of the opposing network.

---

## Problem Statement

Given raw GPS traces (VPD — Vehicle Path Data, HPD — Historical Probe Data) collected over Kosovo, reconstruct the underlying road network as a set of clean LineString geometries, assessed against the HERE Nav Streets reference map.

---

## Pipeline Overview

The solution runs as a **4-phase pipeline** (`run_pipeline.py`):

### Phase 1 — Data Ingestion & Normalization
- Bbox-clipped loading of VPD and HPD CSV files
- Speed filtering to remove pedestrian/cycling traces
- Per-segment heading computation (length-weighted, 8-compass-octant bins)
- Local metric projection (EPSG:32634) for all geometric operations
- Altitude data preserved for downstream Z-level handling

### Phase 2 — Centerline Generation (Kharita-style Clustering)
- Heading-aware incremental clustering of probe sample points
- Directed co-occurrence graph built from trace transitions
- Three-pass edge pruning (support, direction-conflict, transitive)
- Roundabout detection and preservation
- Intersection detection and line splitting
- Segment averaging + turn-preserving smoothing
- Dynamic weighting based on trace quality and sensor confidence

### Phase 3 — Geometry Refinement & Topology Cleanup
- Parallel segment consolidation (Hausdorff-based, conservative thresholds)
- Z-level crossing resolution (splits overlapping grade-separated segments)
- Sharp-angle vertex removal (bird-nesting fix)
- B-spline smoothing with Hausdorff deviation guard
- GPS jitter spike removal (starburst noise)
- Dead-end stub pruning (iterative, NetworkX-based)

### Phase 4 — Final Optimization & Export
- Complex interchange / cloverleaf zone cleanup
- Optional quality-based candidate selection
- Final geometry cleaning and export to GeoPackage (EPSG:4326 + local CRS)

---

## Repository Structure

```
├── run_pipeline.py          # Main pipeline runner (all 4 phases)
├── main.py                  # Standalone data loader + validation entry point
├── src/
│   ├── config.py            # Paths, CRS, bounding box constants
│   ├── pipeline_phase1.py   # Data ingestion & normalization
│   ├── pipeline_phase2.py   # Centerline generation
│   ├── pipeline_phase3.py   # Geometry refinement & topology
│   ├── pipeline_phase4.py   # Final optimization & export
│   ├── algorithms/          # Core algorithm modules
│   │   ├── centerline_utils.py
│   │   ├── curve_smoothing.py
│   │   ├── dynamic_weighting.py
│   │   ├── intersection_detection.py
│   │   ├── roundabout_detection.py
│   │   ├── segment_averaging.py
│   │   ├── topology_builder.py
│   │   └── trajectory_clustering.py
│   ├── evaluation/
│   │   └── metrics.py       # Segmented precision/recall evaluation
│   ├── loaders/             # VPD, HPD, Nav Streets data loaders
│   └── preprocessing/       # Data cleaning & validation
├── data/
│   ├── Kosovo_VPD/          # Vehicle Path Data (GPS traces)
│   ├── Kosovo_HPD/          # Historical Probe Data (weeks 1 & 2)
│   ├── Kosovo_nav_streets/  # HERE Nav Streets ground truth
│   └── nav_streets_attributes_explained.md
├── notebooks/               # Exploratory analysis & prototyping
└── outputs/                 # Pipeline outputs (GeoPackage files)
```

---

## Setup

### Prerequisites

- **Windows 10 / 11**
- Miniconda or Anaconda

### 1. Install Miniconda

Open **PowerShell as Administrator**:

```powershell
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o .\miniconda.exe
start /wait "" .\miniconda.exe /S
del .\miniconda.exe
```

> After installation, close PowerShell and open **Anaconda Prompt**.

### 2. Configure Conda Channels

```powershell
conda config --add channels conda-forge
conda config --set channel_priority strict
```

### 3. Create and Activate the Environment

```powershell
conda env create -f environment.yml
conda activate here-env
```

---

## Running the Pipeline

```powershell
python run_pipeline.py
```

Optional flags:

```powershell
# Limit VPD sample size (faster dev runs)
python run_pipeline.py --sample 5000

# Skip Phase 1 if interim outputs already exist
python run_pipeline.py --skip-phase1

# Tune clustering radius
python run_pipeline.py --cluster-radius-m 15.0
```

The pipeline outputs GeoPackage files to `data/` and prints precision/recall metrics after each phase.

---

## Data

| Dataset | Description |
|---|---|
| **VPD** (Vehicle Path Data) | ~4 GB CSV of GPS traces with path geometry, speed, altitude, quality scores |
| **HPD** (Historical Probe Data) | Two weeks of probe traces over Kosovo |
| **Nav Streets** | HERE reference road network used as ground truth for evaluation |

All data is scoped to a 10 × 10 km bounding box over Kosovo (`EPSG:4326`).

---

## Key Dependencies

- **GeoPandas & Shapely** — Vector geometry operations
- **PyProj** — CRS transformations (EPSG:4326 ↔ EPSG:32634)
- **SciPy** — Spatial indexing (`cKDTree`), B-spline smoothing
- **NetworkX** — Topology graph analysis & stub pruning
- **NumPy / Pandas** — Data processing
- **JupyterLab** — Exploratory notebooks

---

## Troubleshooting

**"Conda is stuck solving the environment..."**
- Press `Ctrl+C`, run `conda clean --all`, then retry.

**"DLL Load Failed" on Windows**
- Conflict between Conda and Pip versions of `shapely`/`fiona`.
- Fix: `pip uninstall shapely fiona` then `conda install -c conda-forge shapely fiona`

**Pipeline runs out of memory on VPD loading**
- Use `--sample 3000` to reduce the VPD sample size.
