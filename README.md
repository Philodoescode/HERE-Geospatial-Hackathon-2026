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
python main.py
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

The repository now includes a Kharita-inspired centerline pipeline adapted for HERE VPD/HPD data:

- `scripts/prepare_navstreet_ground_truth.py`
- `scripts/generate_centerlines.py`
- `scripts/evaluate_centerlines.py`

### 1. Prepare Kosovo navstreet ground truth

```powershell
python scripts/prepare_navstreet_ground_truth.py
```

### 2. Generate centerlines from VPD + HPD

```powershell
python scripts/generate_centerlines.py
```

### 3. Evaluate generated centerlines

```powershell
python scripts/evaluate_centerlines.py
```
