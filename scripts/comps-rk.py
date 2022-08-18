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
import gstools as gs
from scipy import stats

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
var = "data"

# %% [markdown]
# ### Create Grid
# Again we will create a grid that we want to use for the interpolation.
#
# - The grid in the fromate of a dataset is helpful for reprojecting our covariates to match the interpolated grid.
# %%
## define the desired  grid resolution in meters
resolution = 0.25  # grid cell size in meters

## make grid based on dataset bounds and resolution
g_lon = np.arange(
    df["lon"].min() - resolution,
    df["lon"].max() + resolution,
    resolution,
)
g_lat = np.arange(
    df["lat"].min() - resolution,
    df["lat"].max() + resolution,
    resolution,
)

## use salem to create a dataset with the grid.
krig_ds = salem.Grid(
    nxny=(len(g_lon), len(g_lat)),
    dxdy=(resolution, resolution),
    x0y0=(df["lon"].min(), df["lat"].min()),
    proj="EPSG:4326",
    pixel_ref="corner",
).to_dataset()
## print dataset
krig_ds


# %% [markdown]
# ### Covariate
# We will use aersoal optical depth (AOD) as a covariate for universal kriging with specified drift. The data is from the [modis aqua](https://www.nsstc.uah.edu/data/sundar/MODIS_AOD_L3_HRG/) satellite during the datetime of interest
#
# %%
ds = salem.open_xr_dataset(str(data_dir) + f"/elev.americas.5-min.nc").isel(time=0)
ds["lon"] = ds["lon"] - 360


# %% [markdown]
# ## Set up specified drift
# For specified we need satellite derived aod at every aq monitor location and AOD on the same grid we are interpolating.
# %% [markdown]
# ### AOD at AQs location
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
# var_points = ds[var].interp(
#     Longitude=x, Latitude=y, method="linear"
# )
var_points = ds[var].interp(lon=x, lat=y, method="linear")
# print(var_points)
if len(df.index) == len(var_points.values):
    var_points = var_points.values
else:
    raise ValueError("Lenghts dont match")

# var_points[var_points<0] = 0
# var_points[var_points>1] = 1

lat, lon, pm25 = df["lat"], df["lon"], df["PM2.5"]

bins = gs.standard_bins((lat, lon), max_dist=np.deg2rad(8), latlon=True)
bin_c, vario = gs.vario_estimate((lat, lon), pm25, bin_edges=bins, latlon=True)


model = gs.Spherical(latlon=True, rescale=gs.EARTH_RADIUS)
para, pcov, r2 = model.fit_variogram(bin_c, vario, nugget=False, return_r2=True)
ax = model.plot(x_max=bin_c[-1])
ax.scatter(bin_c, vario)
ax.set_xlabel("great circle distance / radians")
ax.set_ylabel("semi-variogram")
fig = ax.get_figure()
# fig.savefig(os.path.join("..", "results", "variogram.pdf"), dpi=300)
print(r2)

###############################################################################

# fit linear regression model for pm25 depending on aod
regress = stats.linregress(var_points, pm25)
trend = lambda x, y: regress.intercept + regress.slope * x

startTime = datetime.now()
dk = gs.krige.Detrended(
    model=model,
    cond_pos=(lat, lon),
    cond_val=pm25.values,
    trend=trend,
)
print(f"RK build time {datetime.now() - startTime}")

###############################################################################
# Now we generate the kriging field, by defining a lat-lon grid that covers
# the whole of Germany. The :any:`Krige` class provides the option to only
# krige the mean field, so one can have a glimpse at the estimated drift.


startTime = datetime.now()
fld_dk = dk((g_lat, g_lon), mesh_type="structured", return_var=False)
print(f"RK exectue time {datetime.now() - startTime}")

RK_pm25 = np.where(fld_dk < 0, 0, fld_dk)

# %% [markdown]
# #### Plot AOD

# %%
# ax = plt.axes(projection=ccrs.Orthographic(-80, 35))
# ax.set_global()
# ds[var].plot(ax=ax, transform=ccrs.PlateCarree())
# ax.coastlines()
# ax.set_extent([-132, -85, 35, 65], crs=ccrs.PlateCarree())

# UK_pm25 = np.where(z < 0, 0, z)

# krig_ds["UK_pm25"] = (("y", "x"), UK_pm25)

# # %% [markdown]
# # ### Plot UK
# # Convert data to polygons to be plot-able on a slippy mapbox. This is not necessary but but :)

# # %%
polygons, values = pixel2poly(g_lon, g_lat, RK_pm25, resolution)
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
    opacity=0.6,
)
fig.update_layout(margin=dict(l=0, r=0, t=30, b=10))
fig.update_traces(marker_line_width=0)
