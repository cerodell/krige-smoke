import context
import numpy as np
import pandas as pd
import xarray as xr
import salem
from datetime import datetime


import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from shapely.geometry import Polygon
import shapely
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
from utils.utils import pixel2poly

from context import data_dir, img_dir
import time

start_time = time.time()

# %%

nlags = 15
variogram_model = "spherical"
frac = 0.1

wesn = [-160.0, -52.0, 32.0, 70.0]  ## BSC Domain
# wesn = [-129.0, -90.0, 40.0, 60.0]  ## Big Test Domain
resolution = 36000  # cell size in meters

era_ds = salem.open_xr_dataset(str(data_dir) + f"/era5-20120716T2200.nc")

dem_ds = salem.open_xr_dataset(str(data_dir) + f"/elev.americas.5-min.nc")
dem_ds["lon"] = dem_ds["lon"] - 360

gov_ds = xr.open_dataset(str(data_dir) + f"/gov_aq.nc")
gov_ds = gov_ds.sel(datetime="2021-07-16T22:00:00")

pa_ds = xr.open_dataset(str(data_dir) + f"/purpleair_north_america.nc")
pa_ds = pa_ds.sel(datetime="2021-07-16T22:00:00")
pa_ds = pa_ds.drop(["PM1.0", "PM10.0", "pressure", "PM2.5_ATM"])

ds = xr.concat([pa_ds, gov_ds], dim="id")

# After droping outliers
ds = ds.where(ds["PM2.5"] < 1000, drop=True)
ds = ds.where(ds["PM2.5"] > 0, drop=True)
mean = ds["PM2.5"].mean()
sd = ds["PM2.5"].std()
sd_ds = ds.where(
    (ds["PM2.5"] > mean - 2 * sd) & (ds["PM2.5"] < mean + 2 * sd), drop=True
)

df_pm25 = sd_ds["PM2.5"].to_dataframe().reset_index()

df_pm25 = df_pm25.loc[
    (df_pm25["lat"] > wesn[2])
    & (df_pm25["lat"] < wesn[3])
    & (df_pm25["lon"] > wesn[0])
    & (df_pm25["lon"] < wesn[1])
]

gpm25 = gpd.GeoDataFrame(
    df_pm25,
    crs="EPSG:4326",
    geometry=gpd.points_from_xy(df_pm25["lon"], df_pm25["lat"]),
).to_crs("EPSG:3347")
gpm25["Easting"], gpm25["Northing"] = gpm25.geometry.x, gpm25.geometry.y
gpm25.head()
# gpm25.to_csv(str(data_dir) + "/obs/gpm25.csv")

# %%


gridx = np.arange(gpm25.bounds.minx.min(), gpm25.bounds.maxx.max(), resolution)
gridy = np.arange(gpm25.bounds.miny.min(), gpm25.bounds.maxy.max(), resolution)


grid_ds = salem.Grid(
    nxny=(len(gridx), len(gridy)),
    dxdy=(resolution, resolution),
    x0y0=(gpm25.bounds.minx.min(), gpm25.bounds.miny.min()),
    proj="epsg:3347",
    pixel_ref="corner",
)

krig_ds = grid_ds.to_dataset()
# era_ds = krig_ds.salem.transform(era_ds)
dem = krig_ds.salem.transform(dem_ds)
dem["data"] = xr.where(dem.data < 0, 0, dem.data)


startTime = datetime.now()
krig = OrdinaryKriging(
    x=gpm25["Easting"],
    y=gpm25["Northing"],
    z=gpm25["PM2.5"],
    variogram_model=variogram_model,
    nlags=nlags,
    # external_drift=dem.data.values,
)
print(f"UK build time {datetime.now() - startTime}")

startTime = datetime.now()
z, ss = krig.execute("grid", gridx, gridy)
print(f"UK execution time {datetime.now() - startTime}")
UK_pm25 = np.where(z < 0, 0, z)


krig_ds.to_netcdf(
    str(data_dir) + f"/test.nc",
    mode="w",
)

krig_ds = salem.open_xr_dataset(str(data_dir) + f"/test.nc")

gridx = np.arange(gpm25.lon.min(), gpm25.lon.max(), 0.1)
gridy = np.arange(gpm25.lat.min(), gpm25.lat.max(), 0.1)


krig_dss = salem.Grid(
    nxny=(len(gridx), len(gridy)),
    dxdy=(0.1, 0.1),
    x0y0=(gpm25.lon.min(), gpm25.lat.min()),
    proj="epsg:4326",
    pixel_ref="corner",
).to_dataset()


test = krig_dss.salem.transform(krig_ds)
