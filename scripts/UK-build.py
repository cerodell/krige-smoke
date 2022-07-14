import context
import numpy as np
import pandas as pd
import xarray as xr
import salem
from datetime import datetime


# %% [markdown]
# # Test
# %%

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

wesn = [-129.0, -90.0, 40.0, 60.0]  ## Big Test Domain
resolution = 10_000  # cell size in meters

era_ds = salem.open_xr_dataset(str(data_dir) + f"/era5-20120716T2200.nc")

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

gpm25_poly = gpd.read_file(str(data_dir) + "/obs/outer_bounds")
gpm25_poly_buff = gpm25_poly.buffer(-80_000)
gpm25_buff = gpd.GeoDataFrame(
    {"geometry": gpd.GeoSeries(gpm25_poly_buff)}, crs=gpm25.crs
)
gpm25_verif = gpd.sjoin(gpm25, gpm25_buff, predicate="within")


gridx = np.arange(gpm25.bounds.minx.min(), gpm25.bounds.maxx.max(), resolution)
gridy = np.arange(gpm25.bounds.miny.min(), gpm25.bounds.maxy.max(), resolution)


grid_ds = salem.Grid(
    nxny=(len(gridx), len(gridy)),
    dxdy=(resolution, resolution),
    x0y0=(gpm25.bounds.minx.min(), gpm25.bounds.miny.min()),
    proj="epsg:3347",
    pixel_ref="corner",
).to_dataset()

era_ds = grid_ds.salem.transform(era_ds)
Angle = np.arctan2(era_ds.v10, era_ds.u10) * (180 / np.pi)


list_ds = []
for i in range(0, 20):
    loopTime = datetime.now()
    gpm25_veriff = gpm25_verif.sample(frac=1).reset_index(drop=True)
    print(i)
    # print(len(gpm25_veriff))
    ds = grid_ds
    random_sample = gpm25_veriff.sample(frac=frac, replace=True, random_state=1)
    gpm25_krig = gpm25[~gpm25.id.isin(random_sample.id)]
    krig = UniversalKriging(
        x=gpm25_krig["Easting"],
        y=gpm25_krig["Northing"],
        z=gpm25_krig["PM2.5"],
        variogram_model=variogram_model,
        nlags=nlags,
        external_drift=Angle.values,
    )

    z, ss = krig.execute("grid", gridx, gridy)
    OK_pm25 = np.where(z < 0, 0, z)
    # gridxx, gridyy = np.meshgrid(gridx, gridy)

    ds.assign_coords({"test": i})
    ds.assign_coords({"ids": np.arange(len(random_sample.id.values))})
    ds["pm25"] = (("y", "x"), OK_pm25)
    ds["random_sample"] = ("ids", random_sample.id.values.astype(str))
    print(f"Loop time {datetime.now() - loopTime}")

    list_ds.append(ds)


final_ds = xr.concat(list_ds, dim="test")
final_ds["random_sample"] = final_ds["random_sample"].astype(str)


def compressor(ds):
    """
    this function compresses datasets
    """
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    return ds, encoding


ds_concat, encoding = compressor(final_ds)
final_ds.to_netcdf(
    str(data_dir)
    + f"/UK-dir-{krig.variogram_model.title()}-{nlags}-{int(frac*100)}.nc",
    encoding=encoding,
    mode="w",
)
