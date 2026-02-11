# HERE-Geospatial-Hackathon

Welcome to the repository for the **Feature Extraction Using GIS Data** hackathon project. This project leverages **LiDAR point clouds**, **Satellite Imagery**, and **Deep Learning** to extract semantic features (buildings, roads, vegetation) from raw geospatial data.

## Prerequisites

- **Windows 10 / 11**
- Administrator access (required only for Miniconda installation)
- Internet connection

---

## Step 1: Install Miniconda (Automated)

Open **PowerShell as Administrator** and run the following commands.
This will:

- Download Miniconda
- Install it silently for the current user
- Remove the installer afterward

```powershell
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o .\miniconda.exe
start /wait "" .\miniconda.exe /S
del .\miniconda.exe
```

> **Important**  
> After installation completes, close PowerShell and open **Anaconda Prompt**.
> The `conda` command will not be available in a normal PowerShell session yet.

---

## Step 2: Configure Global Conda Channels

To ensure packages are installed from trusted sources, configure `conda-forge` as the primary channel.

Run the following inside **Anaconda Prompt**:

```powershell
conda config --add channels conda-forge
conda config --set channel_priority strict
```

This guarantees:

- Consistent builds
- Fewer dependency conflicts
- Faster environment resolution

---

## Step 3: Create the Environment

### 1. Create the environment from the provided file

```powershell
conda env create -f environment.yml
```

### 2. Activate the environment

```powershell
conda activate here-env
```

---

## Verification

To ensure your environment is ready, run:

```powershell
python scripts/preprocess_traces.py --help
```

---

## Environment Summary

After setup, the environment includes:

- **Python 3.10**
- **GDAL & PDAL** - Geospatial data processing
- **GeoPandas & Shapely** - Vector data manipulation
- **Rasterio & PyProj** - Raster & projection tools
- **LASpy & LAZrs** - LAS/LAZ point cloud support
- **Open3D** - 3D point cloud visualization
- **PyTorch & TorchVision** - Deep learning
- **JupyterLab** - Interactive notebooks
- **Scikit-learn & Matplotlib** - ML & visualization

---

## Troubleshooting

**"Conda is stuck solving the environment..."**

- Cancel the process (`Ctrl+C`)
- Run `conda clean --all` and try again

**"DLL Load Failed" on Windows?**

- This usually means a conflict between Conda and Pip versions of `shapely` or `fiona`
- Fix: `pip uninstall shapely fiona` -> `conda install -c conda-forge shapely fiona`

**Environment creation fails?**

- Ensure you have a stable internet connection
- Try running `conda clean --all -y` first
- Make sure conda-forge channel is configured (Step 2)

---

## Centerline Generation (Problem 1)

The repository includes two pluggable map-construction algorithms:

- `kharita` (baseline)
- `roadster` (Roadster-adapted subtrajectory clustering pipeline)

### Core scripts

- `scripts/preprocess_traces.py` (Probe/VPD preprocessing)
- `scripts/prepare_navstreet_ground_truth.py` (Kosovo navstreet preprocessing)
- `scripts/generate_centerlines.py` (map construction)
- `scripts/evaluate_centerlines.py` (metrics + optional plots)

### 1. Preprocess Probe/VPD traces

```powershell
python scripts/preprocess_traces.py `
  --vpd-csv data/Kosovo_VPD/Kosovo_VPD.csv `
  --hpd-csvs data/Kosovo_HPD/XKO_HPD_week_1.csv data/Kosovo_HPD/XKO_HPD_week_2.csv `
  --out-dir outputs/preprocessed `
  --stem kosovo_preprocessed
```

### 2. Prepare Kosovo navstreet ground truth

```powershell
python scripts/prepare_navstreet_ground_truth.py `
  --nav-csv "data/Kosovo's nav streets/Kosovo.csv" `
  --out-dir outputs/ground_truth `
  --stem kosovo_navstreet_ground_truth
```

### 3. Generate centerlines (Roadster)

```powershell
python scripts/generate_centerlines.py `
  --algorithm roadster `
  --vpd-csv data/Kosovo_VPD/Kosovo_VPD.csv `
  --hpd-csvs data/Kosovo_HPD/XKO_HPD_week_1.csv data/Kosovo_HPD/XKO_HPD_week_2.csv `
  --out-dir outputs/centerlines_roadster `
  --stem kosovo_centerlines_roadster
```

### 4. Evaluate against navstreets (+ plots)

```powershell
python scripts/evaluate_centerlines.py `
  --generated outputs/centerlines_roadster/kosovo_centerlines_roadster.gpkg `
  --generated-layer centerlines `
  --ground-truth outputs/ground_truth/kosovo_navstreet_ground_truth.gpkg `
  --ground-truth-layer navstreet `
  --out outputs/evaluation/roadster_metrics.json `
  --compute-topology-metrics `
  --compute-itopo `
  --topology-radii-m 8,15 `
  --compute-hausdorff `
  --return-timing `
  --plots-out-dir outputs/evaluation/plots `
  --plot-stem roadster_eval
```

### Notes

- Roadster adaptation details: `context/roadster_adaptation_notes.md`
- Preprocessing uses source-specific settings (tighter for VPD, more conservative for Probe/HPD).

---

### Kharita Recall Tuning (Problem 1)

Use the recall-oriented tuner with full-bbox/no-GT-clipping evaluation and
precision/size guardrails:

```powershell
python scripts/tune_kharita_recall.py `
  --search random `
  --trials 24 `
  --apply-bbox `
  --no-clip-generated-to-ground-truth `
  --precision-floor 0.90 `
  --max-length-ratio 2.0 `
  --out-dir outputs/tuning_kharita_recall
```

This writes:

- `outputs/tuning_kharita_recall/tuning_results.csv`
- `outputs/tuning_kharita_recall/tuning_best.json`
- `outputs/tuning_kharita_recall/comparison_metrics.csv` (baseline raw vs selected vs selected+turn-smoothing)

### Roadster Tuning Helper

```powershell
python scripts/tune_roadster_params.py `
  --search random `
  --trials 24 `
  --out-dir outputs/tuning_roadster
```
