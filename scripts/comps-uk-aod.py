# %% [markdown]
# # Universal Kriging (UK)

# - Universal kriging is used to interpolate given data with a variable mean, that is determined by a functional drift.
# - This estimator is set to be unbiased by default.
# - This means, that the weights in the kriging equation sum up to 1.
# - Consequently no constant function needs to be given for a constant drift, since the unbiased condition is applied to all given drift functions.

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
# Lets open aersoal optical depth (AOD) data from the modis aqua satellite during the datetime of interest
#
# %%
aod_aqua_ds = salem.open_xr_dataset(str(data_dir) + f"/MYD04.2021197.G10.nc")


# %% [markdown]
# #### Plot ERA5

# %%
aod_aqua_ds["AOD_550_GF_SM"].plot()


# %% [markdown]
# ## Set up specified drift
# lets get satellite derived aod at every aq monitor location

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
var_points = aod_aqua_ds["AOD_550_GF_SM"].interp(
    longitude=x, latitude=y, method="linear"
)
# print(var_points)
if len(df.index) == len(var_points.values):
    var_points = var_points.values
else:
    raise ValueError("Lenghts dont match")


# %% [markdown]
# ### Transform AOD Data
# Now we will transform the aod data to be on the grid we are interpolating too. This is feed in as a specified drift array when exceuting the interpolation.
aod_aqua_ds_T = krig_ds.salem.transform(aod_aqua_ds)
var_array = aod_aqua_ds_T["AOD_550_GF_SM"].values


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
    specified_drift=[var_points],  ## wind direction at aq monitors
)
print(f"UK build time {datetime.now() - startTime}")

# %% [markdown]
# #### Our variogram parameters
# PyKrige will optimize most parameters based on user defined empirical model and the number of bins.
# - I tested several empirical models and bin sizes and found (for this case study) that a spherical model with 15 bins was optimal based on the output statics.
# - NOTE the literature supports spherical for geospatial interpolation applications over other methods.
# %%
plotvariogram(krig)


# %% [markdown]
# ### Execute UK
# Interpolate data to our grid using UK with specified drift. Where the specified drift is the linear correlation of wind direction to pm2.5 at all locations and on the interploated grid for kriging.
# %%
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
    opacity=0.8,
)
fig.update_layout(margin=dict(l=0, r=0, t=30, b=10))
fig.update_traces(marker_line_width=0)
