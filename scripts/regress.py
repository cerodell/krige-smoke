import sys
import context
import salem

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from pykrige.rk import RegressionKriging
from context import data_dir

svr_model = SVR(C=0.1, gamma="auto")
rf_model = RandomForestRegressor(n_estimators=100)
lr_model = LinearRegression(normalize=True, copy_X=True, fit_intercept=False)

models = [svr_model, rf_model, lr_model]


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
# We will use aersoal optical depth (AOD) as a covariate for universal kriging with specified drift. The data is from the [modis aqua](https://www.nsstc.uah.edu/data/sundar/MODIS_AOD_L3_HRG/) satellite during the datetime of interest
#
# %%
aod_aqua_ds = salem.open_xr_dataset(str(data_dir) + f"/MYD04.2021197.G10.nc")


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
var_points = aod_aqua_ds["AOD_550_GF_SM"].interp(
    Longitude=x, Latitude=y, method="linear"
)
# print(var_points)
if len(df.index) == len(var_points.values):
    var_points = var_points.values
else:
    raise ValueError("Lenghts dont match")


lat, lon, pm25 = gpm25["lat"].values, gpm25["lon"].values, gpm25["PM2.5"].values

points = list()
lat_lon = np.array([[e.lat, e.lon] for e in points])
lat_lon = np.column_stack((lat, lon))

p_train, p_test, x_train, x_test, target_train, target_test = train_test_split(
    pm25, var_points, lat_lon, test_size=0.3, random_state=42
)
try:
    housing = fetch_california_housing()
except PermissionError:
    # this dataset can occasionally fail to download on Windows
    sys.exit(0)

# take the first 5000 as Kriging is memory intensive
p = housing["data"][:5000, :-2]
x = housing["data"][:5000, -2:]
target = housing["target"][:5000]

p_train, p_test, x_train, x_test, target_train, target_test = train_test_split(
    p, x, target, test_size=0.3, random_state=42
)

for m in models:
    print("=" * 40)
    print("regression model:", m.__class__.__name__)
    m_rk = RegressionKriging(regression_model=m, n_closest_points=10)
    m_rk.fit(p_train, x_train, target_train)
    # test = m_rk.predict(p_train, x_train)
    print("Regression Score: ", m_rk.regression_model.score(p_test, target_test))
    print("RK score: ", m_rk.score(p_test, x_test, target_test))
