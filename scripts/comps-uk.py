# %% [markdown]
# # Universal Kriging (UK)

# TODO define Universal Kriging


# %%
import context
import salem
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pykrige.uk import UniversalKriging

import plotly.express as px
from datetime import datetime

from utils.utils import pixel2poly, plotvariogram
from context import data_dir

# %% [markdown]
# Open the reformated data with the linear, meter-based, Lambert projection (EPSG:3347). Again this is helpful as lat/lon coordinates are not good for measuring distances which is important for spatial interpolation.

# %%
df = pd.read_csv(str(data_dir) + "/obs/gpm25.csv")
gpm25 = gpd.GeoDataFrame(
    df,
    crs="EPSG:4326",
    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
).to_crs("EPSG:3347")
gpm25["Easting"], gpm25["Northing"] = gpm25.geometry.x, gpm25.geometry.y
gpm25.head()


# %% [markdown]
# ### Create Grid
# Again create a we will create a grid that we want to use for the interpolation.
# -This will be more useful reprojecting era5 gridded data to act as covariances for interpolation
# %%
## define the desired  grid resolution in meters
resolution = 20_000  # grid cell size in meters

## make grid based on dataset bounds and resolution
gridx = np.arange(gpm25.bounds.minx.min(), gpm25.bounds.maxx.max(), resolution)
gridy = np.arange(gpm25.bounds.miny.min(), gpm25.bounds.maxy.max(), resolution)

## use salem to create a dataset with the grid.
krig_ds = salem.Grid(
    nxny=(len(gridx), len(gridy)),
    dxdy=(resolution, resolution),
    x0y0=(gpm25.bounds.minx.min(), gpm25.bounds.miny.min()),
    proj="epsg:3347",
    pixel_ref="corner",
).to_dataset()
## print dataset
krig_ds


# %% [markdown]
# ### ERA5 Data
# Lets open era5 land dataset on the datetime of interest and transform the data to be on our grid for interpolation
#
# %%
era_ds = salem.open_xr_dataset(str(data_dir) + f"/era5-20120716T2200.nc")
era_ds["degrees"] = np.arctan2(era_ds.v10, era_ds.u10) * (180 / np.pi)

era_ds = krig_ds.salem.transform(era_ds)

# %% [markdown]
# #### Plot ERA5

# %%
era_ds["degrees"].plot()


# %% [markdown]
# ##  Setup UK
# %%
nlags = 15
variogram_model = "spherical"

startTime = datetime.now()
krig = UniversalKriging(
    x=gpm25["Easting"],
    y=gpm25["Northing"],
    z=gpm25["PM2.5"],
    drift_terms="regional_linear",
    variogram_model=variogram_model,
    nlags=nlags,
    external_drift=era_ds["degrees"].values,
)
print(f"UK build time {datetime.now() - startTime}")


# %% [markdown]
# ### Execute UK
# Interpolate data to our grid using UK.
# %%
startTime = datetime.now()
z, ss = krig.execute("grid", gridx, gridy)
print(f"UK execution time {datetime.now() - startTime}")
UK_pm25 = np.where(z < 0, 0, z)

krig_ds["UK_pm25"] = (("y", "x"), UK_pm25)

# %% [markdown]
# ### Plot UK
# Convert data to polygons to be plot-able on a slippy mapbox. This is not necessary but but :)

# %%
polygons, values = pixel2poly(gridx, gridy, UK_pm25, resolution)
pm25_model = gpd.GeoDataFrame(
    {"Modelled PM2.5": values}, geometry=polygons, crs="EPSG:3347"
).to_crs("EPSG:4326")

fig = px.choropleth_mapbox(
    pm25_model,
    geojson=pm25_model.geometry,
    locations=pm25_model.index,
    color="Modelled PM2.5",
    color_continuous_scale="jet",
    center={"lat": 50.0, "lon": -110.0},
    zoom=2.5,
    mapbox_style="carto-positron",
    opacity=0.8,
)
fig.update_layout(margin=dict(l=0, r=0, t=30, b=10))
fig.update_traces(marker_line_width=0)
