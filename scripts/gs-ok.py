# %% [markdown]
# # Ordinary Kriging (OK)

# - Ordinary Kriging (OK) is a commonly used geostatistical method.
# - OK provides the best linear unbiased estimates (BLUE), where the estimated value for the point of interest is a weighted linear combination of sampled observations (i.e., the sum of weights is 1) [Matheron1963](http://dx.doi.org/10.2113/gsecongeo.58.8.1246).
# - OK is similar to but more advanced than Inverse distance weighting, as the weight 𝜆𝑖 of OK is estimated by minimizing the variance of the prediction errors.
#   - This is achieved by constructing a semivariogram that models the difference between neighboring values.
# - Compared to non-geostatistical algorithms, the strength of ordinary kriging is its ability to model the spatial structure (variance) of the sampled observations.
# - An assumption of ordinary kriging is data stationarity. That is, the mean of the interpolated variable is constant within the search window, which is often not true. This makes OK unsuitable for interpolation over large domains and often requires data transformation.

# <br>
# <br>
# Thank you to Xinli Cai for this great description of OK in her [master thesis](https://era.library.ualberta.ca/items/92cdc6ae-43fd-453f-91f2-5ff275cf85cd/view/164484ed-e950-408c-8be7-39d3764bdc15/Cai_Xinli_201704_MSc.pdf)
# %%
import context
import salem
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import gstools as gs

import plotly.express as px
from datetime import datetime

from utils.utils import pixel2poly, plotvariogram
from context import data_dir

# %% [markdown]
# Open the reformated data with the linear, meter-based Lambert projection (EPSG:3347). Again this is helpful as lat/lon coordinates are not suitable for measuring distances which is vital for spatial interpolation.

# %%
df = pd.read_csv(str(data_dir) + "/obs/gpm25.csv")
df.head()

# %% [markdown]
# ### Create Grid
# Here, we will create a grid we want to use for the interpolation.
# NOTE we will use salem to create a dataset with the grid. This grid as a xarray dataset will be helpful for the universal kriging when we reproject other gridded data to act as covariances for interpolation.
# %%
## define the desired grid resolution in meters
resolution = 0.2  # grid cell size in meters

## make grid based on dataset bounds and resolution
gridx = np.arange(df["lon"].min(), df["lon"].max() + 1, resolution)
gridy = np.arange(df["lat"].min(), df["lat"].max() + 1, resolution)

## use salem to create a dataset with the grid.
krig_ds = salem.Grid(
    nxny=(len(gridx), len(gridy)),
    dxdy=(resolution, resolution),
    x0y0=(df["lon"].min(), df["lat"].min()),
    proj="epsg:4326",
    pixel_ref="corner",
).to_dataset()
## print dataset
krig_ds


# %% [markdown]
# ### Variogram
# #### variogram overview
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

# #### variogram parameters
# - Three parameters that define a variogram..
#     - sill: the total variance where the empirical model levels off,
#       -  is the sum of the nugget plus the sills of each nested structure.
#     - (effective) range: The distance after which data are no longer correlated.
#       -  About the distance where the variogram levels off to the sill.
#     - nugget: Related to the amount of short range variability in the data.
#        - Choose a value for the best fit with the first few empirical variogram points.
#        -  A nugget that's large relative to the sill is problematic and could indicate too much noise and not enough spatial correlation.


#### variogram statistics
# A good model should result in
#   - Q1 close to zero

#   - Q2 close to one
#   - cR as small as possible.
# TODO define above stats variables.

# %% [markdown]
# #### Our variogram parameters
# PyKrige will optimize most parameters based on user defined empirical model and the number of bins.
# - I tested several empirical models and bin sizes and found (for this case study) that a spherical model with 15 bins was optimal based on the output statics.
# - NOTE the literature supports spherical for geospatial interpolation applications over other methods.
# %%

bins = gs.standard_bins((df["lat"], df["lon"]), max_dist=np.deg2rad(8), latlon=True)
bin_c, vario = gs.vario_estimate(
    (df["lat"], df["lon"]), df["PM2.5"], bin_edges=bins, latlon=True
)


model = gs.Spherical(latlon=True, rescale=gs.EARTH_RADIUS)
model.fit_variogram(bin_c, vario, nugget=False)
ax = model.plot("vario_yadrenko", x_max=bin_c[-1])
ax.scatter(bin_c, vario)
ax.set_xlabel("great circle distance / radians")
ax.set_ylabel("semi-variogram")
fig = ax.get_figure()

print(model)


# %% [markdown]
# ##  Setup OK
# %%

startTime = datetime.now()
ok = gs.krige.Ordinary(
    model=model,
    cond_pos=(df["lat"], df["lon"]),
    cond_val=df["PM2.5"],
)
print(f"ok build time {datetime.now() - startTime}")


# %% [markdown]
# ### Execute OK
# Interpolate data to our grid using OK.
# %%
startTime = datetime.now()
fld_ok = ok((gridy, gridx), mesh_type="structured", return_var=False)
print(f"ok exectue time {datetime.now() - startTime}")
fld_ok = np.where(fld_ok < 0, 0, fld_ok)
krig_ds["fld_ok"] = (("y", "x"), fld_ok)

# %% [markdown]
# ### Plot OK Modelled PM2.5
# Convert data to polygons to be plot-able on a slippy mapbox. The conversion is not necessary, just fun to plot on a slippy map :)

# %%
polygons, values = pixel2poly(gridx, gridy, fld_ok, resolution)


pm25_model = gpd.GeoDataFrame(
    {"Modelled PM2.5": values}, geometry=polygons, crs="EPSG:4326"
)

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
fig.show()


# %% [markdown]
# ### Onto Universal Kriging...
