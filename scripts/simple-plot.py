import context

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors
from context import data_dir


ds = xr.open_dataset(str(data_dir) + f"/purpleair3.nc")
ds = ds.isel(datetime=0)
lats = ds["lat"].values
lons = ds["lon"].values
data = ds["PM2.5"].values

## bring in state/prov boundaries
states_provinces = cfeature.NaturalEarthFeature(
    category="cultural",
    name="admin_1_states_provinces_lines",
    scale="50m",
    facecolor="none",
)

# create fig and axes using intended projection
fig = plt.figure(figsize=(12, 10))
data_crs = ccrs.PlateCarree()
ax = fig.add_subplot(1, 1, 1, projection=data_crs)
ax.add_feature(states_provinces, linewidth=0.5, edgecolor="black", zorder=10)
ax.add_feature(cfeature.BORDERS, zorder=10, lw=0.7)
ax.add_feature(cfeature.COASTLINE, zorder=10, lw=0.7)
fig.tight_layout()
CS = ax.scatter(lons, lats, c=data, cmap="jet")
cbar = fig.colorbar(CS)
