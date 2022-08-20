# %% [markdown]
# # Ordinary Kriging (OK)
# Ordinary kriging (OK) is the most widely used kriging method. It is a linear unbiased estimators since error mean is equal to zero. In OK, local mean is filtered from the linear estimator by forcing the kriging weights to sum to 1. The OK is usually preferred to simple kriging because it requires neither knowledge nor stationarity of mean over the entire area
# $$
# \begin{array}{l}Z_{O K}{ }^{*}(u)=\sum_{\alpha=1}^{n(u)} \lambda_{\alpha}^{O K}(u) Z\left(u_{\alpha}\right) \text { with } \sum_{\alpha=1}^{n(u)} \lambda_{\alpha}^{O K}=1\end{array}
# $$
# - Where is $\lambda_{\alpha}^{O K}(u)$  weight assigned to the known variables at location  $\left(u_{a}\right)$ and $n(u)$ is the number of measured values used in estimation of the neighborhood of $u$.
#
# - An assumption of ordinary kriging is data stationarity. That is, the mean of the interpolated variable is constant within the search window, which is often not true. This makes OK less suitable for interpolation over large domains and often requires data transformation.
#
# <div class="alert alert-info">
# Note
#
# The definition above come from a [geospatial data science course](https://zia207.github.io/geospatial-r-github.io/index.html) created by [Prof. Zia Ahmed](https://www.buffalo.edu/renew/about-us/leadership/zia-ahmed.html) at The State of New York University at Buffalo.
#
# - Thanks Prof. Zia Ahmed for the great resource!
# </div>
# %% [markdown]
# Load python modules
# %%
import context
import salem
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pykrige.ok import OrdinaryKriging

import plotly.express as px
from datetime import datetime

from utils.utils import pixel2poly, plotvariogram
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
# ## Create Grid
# Here, we will create a grid we want to use for the interpolation.
# NOTE we will use salem to create a dataset with the grid. This grid as a xarray dataset will be helpful for the universal kriging when we reproject other gridded data to act as covariances for interpolation.
# %%
## define the desired grid resolution in meters
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
# ##  Setup OK
# %%
nlags = 15
variogram_model = "spherical"

startTime = datetime.now()
krig = OrdinaryKriging(
    x=gpm25["Easting"],
    y=gpm25["Northing"],
    z=gpm25["PM2.5"],
    variogram_model=variogram_model,
    # enable_statistics=True,
    nlags=nlags,
)
print(f"OK build time {datetime.now() - startTime}")


# %% [markdown]
# ## Variogram
# A variogram is defined as the variance of the difference between field values at two locations.

# ### Overview
# - Graphical representation of spatial autocorrelation.
# - Shows a fundamental principle of geography: closer things are more alike than things farther apart
# - Its created by calculating the difference squared between the values of the paired locations
#   - paired locations are binned by the distance apart
# - An empirical model is fitted to the binned (paired locations) to describe the likeness of data at a distance.
# - Type of empirical models
#    - Circular
#    - Spherical
#    - Exponential
#    - Gaussian
#    - Linear
#  - The fitted model is applied in the interpolation process by forming (kriging) weights for the predicted areas.
#
# ### Parameters
# - Three parameters that define a variogram..
#     - sill: the total variance where the empirical model levels off,
#       -  is the sum of the nugget plus the sills of each nested structure.
#     - (effective) range: The distance after which data are no longer correlated.
#       -  About the distance where the variogram levels off to the sill.
#     - nugget: Related to the amount of short range variability in the data.
#        - Choose a value for the best fit with the first few empirical variogram points.
#        -  A nugget that's large relative to the sill is problematic and could indicate too much noise and not enough spatial correlation.
# %% [markdown]
# ###  Statistics
# A good model should result in:
#
# - Q1 close to zero
#
# - Q2 close to one
#
# - cR as small as possible.
#
# - TODO define above stats variables.
#
# #### Our variogram parameters
# PyKrige will optimize most parameters based on user defined empirical model and the number of bins.
#
#  - I tested several empirical models and bin sizes and found (for this case study) that a spherical model with 15 bins was optimal based on the output statics.
#
# - The literature supports spherical for geospatial interpolation applications over other methods.
# %%
plotvariogram(krig)


# %% [markdown]
# ## Execute OK
# Interpolate data to our grid using OK.
# %%
startTime = datetime.now()
z, ss = krig.execute("grid", gridx, gridy)
print(f"OK execution time {datetime.now() - startTime}")
OK_pm25 = np.where(z < 0, 0, z)

# krig_ds["OK_pm25"] = (("y", "x"), OK_pm25)

# %% [markdown]
# ## Plot OK Modelled PM2.5
# Convert data to polygons to be plot-able on a slippy mapbox. The conversion is not necessary, just fun to plot on a slippy map :)

# %%
polygons, values = pixel2poly(gridx, gridy, OK_pm25, resolution)
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
fig.show()


# %% [markdown]
# ## Onto Universal Kriging...
