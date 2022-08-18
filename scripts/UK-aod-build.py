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
frac = 0.50
var = "AOD_550_GF_SM"
wesn = [-129.0, -90.0, 40.0, 60.0]  ## Big Test Domain
resolution = 10_000  # cell size in meters

aod_aqua_ds = salem.open_xr_dataset(str(data_dir) + f"/MYD04.2021197.G10.nc")


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


# gridx = np.arange(gpm25.bounds.minx.min(), gpm25.bounds.maxx.max(), resolution)
# gridy = np.arange(gpm25.bounds.miny.min(), gpm25.bounds.maxy.max(), resolution)


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


def era_points(df):
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
    var_points = aod_aqua_ds[var].interp(Longitude=x, Latitude=y, method="linear")
    # print(var_points)
    if len(df.index) == len(var_points.values):
        var_points = var_points.values
    else:
        raise ValueError("Lenghts dont match")
    return var_points


# era_ds_T = krig_ds.salem.transform(era_ds)
aod_aqua_ds_T = krig_ds.salem.transform(aod_aqua_ds)
var_array = aod_aqua_ds_T[var].values


def compressor(ds):
    """
    this function compresses datasets
    """
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    return ds, encoding


for frac in [0.1, 0.3, 0.5]:
    print(f"looping {int(frac*100)}")
    random_sample_df = pd.read_csv(
        str(data_dir) + f"/random-samples-{int(frac*100)}.csv", index_col=0
    )
    list_ds, random_ids_list = [], []
    for i in range(0, 10):
        loopTime = datetime.now()

        ds = krig_ds
        # gpm25_veriff = gpm25_verif.sample(frac=1).reset_index(drop=True)
        # random_sample = gpm25_veriff.sample(frac=frac, replace=True, random_state=1)
        random_ids = random_sample_df[str(i)].values
        # print(random_ids)
        gpm25_krig = gpm25.loc[~gpm25.id.isin(random_ids)]
        print(f"Random Sample index 0 {random_ids[0]}")

        var_points = era_points(gpm25_krig)
        startTime = datetime.now()
        krig = UniversalKriging(
            x=gpm25_krig[
                "Easting"
            ].values,  ## x location of aq monitors in lambert conformal
            y=gpm25_krig[
                "Northing"
            ].values,  ## y location of aq monitors in lambert conformal
            z=gpm25_krig[
                "PM2.5"
            ].values,  ## measured PM 2.5 concentrations at locations
            drift_terms=["external_Z"],
            # drift_terms=["specified"],
            variogram_model=variogram_model,
            nlags=nlags,
            external_drift=var_array,  ## 2d array of dem used for external drift
            external_drift_x=gridx,  ## x coordinates of 2d dem data file in lambert conformal
            external_drift_y=gridy,  ## y coordinates of 2d dem data file in lambert conformal
            # specified_drift=[var_points],  ## elevation of aq monitors
        )
        print(f"UK build time {datetime.now() - startTime}")

        startTime = datetime.now()
        # z, ss = krig.execute("grid", gridx, gridy, specified_drift_arrays=[var_array])
        z, ss = krig.execute("grid", gridx, gridy)
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

    ds_concat, encoding = compressor(final_ds)
    final_ds.to_netcdf(
        str(data_dir)
        + f"/UK-aod-ex-{krig.variogram_model.title()}-{nlags}-{int(frac*100)}.nc",
        encoding=encoding,
        mode="w",
    )
