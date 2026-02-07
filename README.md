# HERE Hackathon

## Prerequisites

- **Windows 10 / 11**
- Administrator access (required only for installation)
- Internet connection

## 🚀 Step 1: Install Miniconda (Automated)

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

> ⚠️ **Important**  
> After installation completes, close PowerShell and open **Anaconda Prompt**.
> The `conda` command will not be available in a normal PowerShell session yet.

## 🚀 Step 2: Configure Global Conda Channels

To ensure everyone installs packages from the same trusted source, configure
`conda-forge` as the primary channel.

Run the following inside **Anaconda Prompt**:

```powershell
conda config --add channels conda-forge
conda config --set channel_priority strict
```

This guarantees:

- Consistent builds
- Fewer dependency conflicts
- Faster resolution with `mamba`

## 🚀 Step 3: Create the `geo` Environment

### 1️. Create and activate the environment

```powershell
conda create -n geo python=3.7.9 -y
conda activate geo
```

### 2️. Pin the Python version (important)

This prevents Python from being upgraded accidentally during installs.

```powershell
echo python ==3.7.9 >> "$env:CONDA_PREFIX\conda-meta\pinned"
```

### 3️. Install **Mamba** (faster dependency solver)

```powershell
conda install -c conda-forge mamba -y
```

### 4. Install the Geospatial Stack

```powershell
mamba install geopandas laspy laszip pdal python-pdal shapely rasterio pyproj notebook ipykernel -y
```

### 5. Install Python bindings for compression & point cloud tools

```powershell
pip install laszip pptk
```

## Environment Summary

After setup, the environment includes:

- **Python 3.7.9 (pinned)**
- **GeoPandas**
- **PDAL + python-pdal**
- **LAS / LAZ support**
- **Raster & projection tools**
- **Fast dependency resolution via Mamba**