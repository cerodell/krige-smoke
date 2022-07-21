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
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging


import plotly.express as px
from datetime import datetime

from utils.utils import pixel2poly, plotvariogram
from context import data_dir

# %% [markdown]
# Open the reformated data with the linear, meter-based Lambert projection (EPSG:3347). Again this is helpful as lat/lon coordinates are not suitable for measuring distances which is vital for spatial interpolation.

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


era_ds = salem.open_xr_dataset(str(data_dir) + f"/era5-20120716T2200.nc")
era_ds["degrees"] = np.arctan2(era_ds.v10, era_ds.u10) * (180 / np.pi)

era_ds = krig_ds.salem.transform(era_ds)

# %% [markdown]
# ##  Setup OK
# %%
nlags = 15
variogram_model = "spherical"

startTime = datetime.now()
krig_ok = OrdinaryKriging(
    x=gpm25["Easting"],
    y=gpm25["Northing"],
    z=gpm25["PM2.5"],
    variogram_model=variogram_model,
    # enable_statistics=True,
    nlags=nlags,
)
print(f"OK build time {datetime.now() - startTime}")


startTime = datetime.now()
krig_uk = UniversalKriging(
    x=gpm25["Easting"],
    y=gpm25["Northing"],
    z=gpm25["PM2.5"],
    drift_terms="regional_linear",
    # variogram_model=variogram_model,
    nlags=nlags,
    # external_drift=era_ds["degrees"].values,
)
print(f"UK build time {datetime.now() - startTime}")

startTime = datetime.now()
krig_uk2 = UniversalKriging(
    x=gpm25["Easting"],
    y=gpm25["Northing"],
    z=gpm25["PM2.5"],
    drift_terms="regional_linear",
    variogram_model=variogram_model,
    nlags=nlags,
    # external_drift=era_ds["degrees"].values,
)
print(f"UK build time {datetime.now() - startTime}")

# %%
startTime = datetime.now()
z, ss = krig_ok.execute("grid", gridx, gridy)
print(f"OK execution time {datetime.now() - startTime}")
OK_pm25_ok = np.where(z < 0, 0, z)


startTime = datetime.now()
z, ss = krig_uk.execute("grid", gridx, gridy)
print(f"OK execution time {datetime.now() - startTime}")
OK_pm25_uk = np.where(z < 0, 0, z)


startTime = datetime.now()
z, ss = krig_uk2.execute("grid", gridx, gridy)
print(f"OK execution time {datetime.now() - startTime}")
OK_pm25_uk2 = np.where(z < 0, 0, z)

krig_ds["OK_pm25"] = (("y", "x"), OK_pm25_uk2 - OK_pm25_uk)
krig_ds["OK_pm25"].plot()
