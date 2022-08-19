# %% [markdown]
# # Regression Kriging (RK)
# Regression kriging (RK) mathematically equivalent to the regression kriging or kriging with external drift, where auxiliary predictors are used directly to solve the kriging weights. Regression kriging combines a regression model with simple kriging of the regression residuals. The experimental variogram of residuals is first computed and modeled, and then simple kriging (SK) is applied to the residuals to give the spatial prediction of the residuals.

# $$
# \begin{array}{l}Z_{R K}^{*}(u)=m_{R K}^{*}(u)+\sum_{\alpha=1}^{n(u)} \lambda_{\alpha}^{R K}(u) R\left(u_{\alpha}\right)\end{array}
# $$

# - Where $m^{*} R K(u \alpha)$ is the regression estimate for location $u$ and $R(u \alpha)$ are the residuals $[R(u \alpha)-m(u \alpha)]$ of the observed locations, $n(u)$.


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
import gstools as gs
from scipy import stats

import plotly.express as px
from datetime import datetime

from utils.utils import pixel2poly, plotvariogram
from context import data_dir

import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Load Data
# Open data but leave in an earth latitude/longitude coordinate system (EPSG:4326).
#
# - We are not using the lambert conformal at this time because I struggled to implement Regression Kriging in the [PyKrig](https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/index.html) python package. Instead, we are using the [GStools](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/) python packaged.
#
#  - With the [GStools](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/) Packaged, I had issues working in lambert conformal
# %%
df = pd.read_csv(str(data_dir) + "/obs/gpm25.csv")
lat, lon, pm25 = df["lat"], df["lon"], df["PM2.5"]
var = "data"

# %% [markdown]
# ### Create Grid
#  We will create a grid that we want to use for the interpolation.
#
# %%
## define the desired  grid resolution in degrees
resolution = 0.25  # grid cell size in degress

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
# ### Trend model

# - For regression krigging, we solve the residuals based on a trend model fitted on the relationship of PM2.5 to Elevation.
#
# - The trend in this case is define by a simple linear model regression of PM2.5 and elevation
#
# - After the linear model regression, we will apply simple kriging (SK) to the residuals.
#
# - Note, we obtain elevation at each sensor location from a [digital elevation model](http://research.jisao.washington.edu/data_sets/elevation/) (dem).
#
# %% [markdown]
# ### Open DEM Model

# %%
ds = salem.open_xr_dataset(str(data_dir) + f"/elev.americas.5-min.nc").isel(time=0)
ds["lon"] = ds["lon"] - 360

# %% [markdown]
# #### Plot dem

# %%
ax = plt.axes(projection=ccrs.Orthographic(-80, 35))
ax.set_global()
ds[var].plot(
    ax=ax, transform=ccrs.PlateCarree(), levels=np.arange(0, 3100, 100), cmap="terrain"
)
ax.coastlines()
ax.set_extent([-132, -85, 35, 65], crs=ccrs.PlateCarree())


# %% [markdown]
# #### DEM at AQs location
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
var_points = ds[var].interp(lon=x, lat=y, method="linear")
# print(var_points)
if len(df.index) == len(var_points.values):
    var_points = var_points.values
else:
    raise ValueError("Lengths dont match")

# %% [markdown]
# ### Fit trend
# Fit linear regression model for pm25 depending on dem and plot
# %%
regress = stats.linregress(var_points, pm25)
trend = lambda x, y: regress.intercept + regress.slope * x

plt.plot(var_points, pm25, "o", label="original data")
plt.plot(
    var_points, regress.intercept + regress.slope * var_points, "r", label="fitted line"
)
plt.legend()
plt.show()
# %% [markdown]
# There is clearly no linear trend between elevation and PM2.5 concentration for our case study. one could test different variables, such as AOD in the UK example before, or use more sophisticated machine learning models. However, we will attempt that for this project. Instead, we will test RK with this poorly-fitted linear regression model to show how RK works.
# %% [markdown]
# ### Variogram
# Make Variogram model in GSTool using a Spherical
# %%
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


# %% [markdown]
# ### Build Krige  Model

# %%
startTime = datetime.now()
dk = gs.krige.Detrended(
    model=model,
    cond_pos=(lat, lon),
    cond_val=pm25.values,
    trend=trend,
)
print(f"RK build time {datetime.now() - startTime}")


# %% [markdown]
# ### Exectue Krige

# %%
startTime = datetime.now()
fld_dk = dk((g_lat, g_lon), mesh_type="structured", return_var=False)
print(f"RK exectue time {datetime.now() - startTime}")

RK_pm25 = np.where(fld_dk < 0, 0, fld_dk)


# %% [markdown]
# ### Plot RK
# Convert data to polygons to be plot-able on a slippy mapbox. This is not necessary but but :)

# %%
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
