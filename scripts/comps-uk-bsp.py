# %% [markdown]
# # Universal Kriging (UK)
# Universal Kriging (UK) is a variant of the Ordinary Kriging under non-stationary condition where mean differ in a deterministic way in different locations (local trend or drift), while only the variance is constant. This second-order stationarity (“weak stationarity”) is often a pertinent assumption with environmental exposures. In UK, usually first trend is calculated as a function of the coordinates and then the variation in what is left over (the residuals) as a random field is added to trend for making final prediction.
#
# $$
# \begin{aligned} Z\left(s_{i}\right) &=m\left(s_{i}\right)+e\left(s_{i}\right) \\ Z(\vec{x}) &=\sum_{k=0}^{K} \beta_{k} f_{k}(\vec{x})+\varepsilon(\vec{x}) \end{aligned}
# $$
#
#
# - Where the $f_{k}$ are some global functions of position  $\vec{x}$  and the  $\beta_{k}$ are the coefficients.
#
# - The $f$ are called base functions. The  $\varepsilon(\vec{x})$  is the spatially-correlated error, which is modelled as before, with a variogram, but now only considering the residuals, after the global trend is removed.

# %% [markdown]
# Load python modules
# %%
import context
import salem
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from pykrige.uk import UniversalKriging

import plotly.express as px
from datetime import datetime

from utils.utils import pixel2poly, plotvariogram, cfcompliant
from context import data_dir

# %% [markdown]
# ## Load Data
# Open the reformated data with the linear, meter-based Lambert projection (EPSG:3347).
# - Again this is helpful as lat/lon coordinates are less suitable for measuring distances which is important for spatial interpolation.
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
# Again we will create a grid that we want to use for the interpolation.
#
# - The grid in the fromate of a dataset is helpful for reprojecting our covariates to match the interpolated grid.
# %%
## define the desired  grid resolution in meters
resolution = 20_000  # grid cell size in meters

## make grid based on dataset bounds and resolution
gridx = np.arange(
    gpm25.bounds.minx.min() - resolution,
    gpm25.bounds.maxx.max() + resolution,
    resolution,
)
gridy = np.arange(
    gpm25.bounds.miny.min() - resolution,
    gpm25.bounds.maxy.max() + resolution,
    resolution,
)

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
# ### Covariate
# We will use the Bluesky Canada Smoke Forecast (BSC) as a covariate for universal kriging with specified drift. The data is from the [firesmoke.ca](https://firesmoke.ca)
#
# %%

ds = salem.open_xr_dataset(str(data_dir) + f"/dispersion1.nc")


# %% [markdown]
# ## Set up specified drift
# For specified we need satellite derived BSC PM2.5 at every aq monitor location and BSC PM2.5 on the same grid we are interpolating.
# %% [markdown]
# ### BSC PM2.5 at AQs location
# %%

y = xr.DataArray(
    np.array(df["lat"]),
    dims="ids",
    coords=dict(ids=df.id.values),
)
x = xr.DataArray(
    np.array(df["lon"]),
    dims="ids",
    coords=dict(ids=df.id.values),
)
var_points = ds["pm25"].interp(x=x, y=y, method="linear")
# print(var_points)
if len(df.index) == len(var_points.values):
    var_points = var_points.values
else:
    raise ValueError("Lengths dont match")


# %% [markdown]
# ### BSC PM2.5 Data on grid
# Now we will transform the BSC PM2.5 data to be on the grid we are interpolating too. This is feed in as a specified drift array when executing the interpolation.
# %%
ds_T = krig_ds.salem.transform(ds)
var_array = ds_T["pm25"].values

# %% [markdown]
# #### Plot BSC PM2.5

# %%
ax = plt.axes(projection=ccrs.Orthographic(-80, 35))
ax.set_global()
ds["pm25"].plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    levels=[0, 5, 10, 20, 40, 80, 160, 300, 600],
    cmap="Reds",
)
ax.coastlines()
ax.set_extent([-132, -85, 35, 65], crs=ccrs.PlateCarree())


# %% [markdown]
# ##  Setup UK
# %%
nlags = 15
variogram_model = "spherical"


startTime = datetime.now()
krig = UniversalKriging(
    x=gpm25["Easting"],  ## x location of aq monitors in lambert conformal
    y=gpm25["Northing"],  ## y location of aq monitors in lambert conformal
    z=gpm25["PM2.5"],  ## measured PM 2.5 concentrations at locations
    drift_terms=["specified"],
    variogram_model=variogram_model,
    nlags=nlags,
    specified_drift=[var_points],  ## BSC PM2.5 at aq monitors
)
print(f"UK build time {datetime.now() - startTime}")

# %% [markdown]
# #### Our variogram parameters
# PyKrige will optimize most parameters based on user defined empirical model and the number of bins.
#
# - I tested several empirical models and bin sizes and found (for this case study) that a spherical model with 15 bins was optimal based on the output statics.
#
#  - The literature supports spherical for geospatial interpolation applications over other methods.
# %%
plotvariogram(krig)


# %% [markdown]
# ### Execute UK
# Interpolate data to our grid using UK with specified drift. Where the specified drift is the linear correlation of BSC PM2.5 to PM2.5 at all locations and on the interploated grid for kriging.
# %%
var_array[var_array > np.max(var_points)] = np.max(var_points) + 20

startTime = datetime.now()
z, ss = krig.execute("grid", gridx, gridy, specified_drift_arrays=[var_array])
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
    opacity=0.6,
)
fig.update_layout(margin=dict(l=0, r=0, t=30, b=10))
fig.update_traces(marker_line_width=0)
