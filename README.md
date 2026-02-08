# HERE-Geospatial-Hackathon 🌍📍

Welcome to the repository for the **Feature Extraction Using GIS Data** hackathon project. This project leverages **LiDAR point clouds**, **Satellite Imagery**, and **Deep Learning** to extract semantic features (buildings, roads, vegetation) from raw geospatial data.

## 🛠️ Environment Setup (Crucial!)

**Do not try to install everything at once.** Geospatial libraries (`GDAL`, `PDAL`) and Deep Learning libraries (`PyTorch`) often conflict. Follow this **Strict Hybrid Installation Guide** to set up the environment without errors.

### **Prerequisites**

* **Miniconda** (Recommended) or Anaconda installed.
* **Windows OS** (Commands below are optimized for Windows).

### **Step-by-Step Installation**

**1. Open your Terminal (Anaconda Prompt) and clean your cache:**

```bash
conda clean --all -y

```

**2. Create the Skeleton Environment:**
We use Python 3.10 as it is the most stable for current GIS tools.

```bash
conda create -n here-env python=3.10 -y

```

**3. Activate the Environment:**
*You must do this before proceeding!*

```bash
conda activate here-env

```

**4. Install Heavy Binaries via Conda (The "Hard" Part):**
We install only the complex C++ dependencies here to avoid solver hangs.

```bash
conda install -c conda-forge gdal pdal rasterio fiona -y

```

**5. Install Python Libraries via Pip (The "Fast" Part):**
Once the binaries are set, we use pip for the rest. This is faster and prevents conflicts.

```bash
# Core GIS & 3D tools
pip install geopandas laspy[lazrs] open3d shapely scikit-learn matplotlib jupyterlab

# Deep Learning (PyTorch) - This may take a few minutes to download
pip install torch torchvision

```

---

## ✅ Verification

To ensure your environment is ready, run this quick test in your terminal:

```bash
python -c "import gdal; import pdal; import laspy; import torch; print('✅ SUCCESS: All systems go for here-env!')"

```

If you see `✅ SUCCESS`, you are ready to code.

---

## 🆘 Troubleshooting

**"Conda is stuck solving the environment..."**

* **STOP immediately.** You are likely trying to install `gdal` and `pip` packages in one command.
* Cancel the process (`Ctrl+C`).
* Run `conda clean --all` and restart from Step 2 above.

**"DLL Load Failed" on Windows?**

* This usually means a conflict between Conda and Pip versions of `shapely` or `fiona`.
* Fix: `pip uninstall shapely fiona` -> `conda install -c conda-forge shapely fiona`.
