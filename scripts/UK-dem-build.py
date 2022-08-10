import context
import numpy as np
import pandas as pd
import xarray as xr
import salem
from datetime import datetime


import geopandas as gpd
import plotly.graph_objects as go
from pykrige.uk import UniversalKriging


from context import data_dir, img_dir
import time

start_time = time.time()

# %%

nlags = 15
variogram_model = "spherical"
frac = 0.10


wesn = [-129.0, -90.0, 40.0, 60.0]  ## Big Test Domain
resolution = 10_000  # cell size in meters

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


def dem_points(df):
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
    var_points = dem_ds.data.interp(lon=x, lat=y, method="linear").values[0, :]
    if len(df.index) == len(var_points):
        pass
    else:
        raise ValueError("Lenghts dont match")
    return var_points
    # df["dem"] = dem_points.data[0, :]
    # return df["dem"].values


# era_ds_T = grid_ds.salem.transform(era_ds)
dem_ds_T = krig_ds.salem.transform(dem_ds)
dem_array = dem_ds_T.data.values[0, :, :].T


gpm25_verif = gpm25
list_ds, random_ids_list = [], []
for i in range(0, 10):
    loopTime = datetime.now()

    ds = krig_ds
    gpm25_veriff = gpm25_verif.sample(frac=1).reset_index(drop=True)
    random_sample = gpm25_veriff.sample(frac=frac, replace=True, random_state=1)
    random_ids = random_sample.id.values
    gpm25_krig = gpm25.loc[~gpm25.id.isin(random_sample.id)]
    print(f"Random Sample index 0 {random_ids[0]}")
    dem = dem_points(gpm25_krig)
    startTime = datetime.now()
    krig = UniversalKriging(
        x=gpm25_krig["Easting"],  ## x location of aq monitors in lambert conformal
        y=gpm25_krig["Northing"],  ## y location of aq monitors in lambert conformal
        z=gpm25_krig["PM2.5"],  ## measured PM 2.5 concentrations at locations
        drift_terms=["specified"],
        # drift_terms=["external_Z", "specified"],
        variogram_model=variogram_model,
        nlags=nlags,
        # external_drift=dem_array,  ## 2d array of dem used for external drift
        # external_drift_x=gridx,  ## x coordinates of 2d dem data file in lambert conformal
        # external_drift_y=gridy,  ## y coordinates of 2d dem data file in lambert conformal
        specified_drift=[dem],  ## elevation of aq monitors
    )
    print(f"UK build time {datetime.now() - startTime}")

    startTime = datetime.now()
    z, ss = krig.execute("grid", gridx, gridy, specified_drift_arrays=[dem_array])
    UK_pm25 = np.where(z < 0, 0, z)
    print(f"UK execute time {datetime.now() - startTime}")

    ds.assign_coords({"test": i})
    ds.assign_coords({"ids": np.arange(len(random_ids))})
    ds["pm25"] = (("y", "x"), UK_pm25)
    random_ids_list.append(random_ids.astype(str))

    list_ds.append(ds)
    print(f"Loop {i} time {datetime.now() - loopTime}")


final_ds = xr.concat(list_ds, dim="test")
final_ds["random_sample"] = (("test", "ids"), np.stack(random_ids_list))
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
    + f"/UK-dem-sp-{krig.variogram_model.title()}-{nlags}-{int(frac*100)}.nc",
    encoding=encoding,
    mode="w",
)