from osgeo import gdal
import pdal
import laspy
import torch

print('SUCCESS: All systems go for here-env!')

import pandas as pd
# Read CSV and convert to Parquet
pd.read_csv("data/Kosovo_VPD/Kosovo_VPD.csv").to_parquet("kosvo_vpd.parquet", engine="pyarrow")
