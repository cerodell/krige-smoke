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
frac = 0.50


wesn = [-129.0, -90.0, 40.0, 60.0]  ## Big Test Domain
resolution = 10_000  # cell size in meters

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
gpm25.to_csv(str(data_dir) + "/obs/gpm25.csv")

# # %%

# gpm25_poly = gpd.read_file(str(data_dir) + "/obs/outer_bounds")
# gpm25_poly_buff = gpm25_poly.buffer(-80_000)
# gpm25_buff = gpd.GeoDataFrame(
#     {"geometry": gpd.GeoSeries(gpm25_poly_buff)}, crs=gpm25.crs
# )
# gpm25_verif = gpd.sjoin(gpm25, gpm25_buff, predicate="within")

# %%


gridx = np.arange(gpm25.bounds.minx.min(), gpm25.bounds.maxx.max(), resolution)
gridy = np.arange(gpm25.bounds.miny.min(), gpm25.bounds.maxy.max(), resolution)


grid_ds = salem.Grid(
    nxny=(len(gridx), len(gridy)),
    dxdy=(resolution, resolution),
    x0y0=(gpm25.bounds.minx.min(), gpm25.bounds.miny.min()),
    proj="epsg:3347",
    pixel_ref="corner",
).to_dataset()


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

        ds = grid_ds
        # gpm25_veriff = gpm25_verif.sample(frac=1).reset_index(drop=True)
        # random_sample = gpm25_veriff.sample(frac=frac, replace=True, random_state=1)
        random_ids = random_sample_df[str(i)].values
        # print(random_ids)
        gpm25_krig = gpm25.loc[~gpm25.id.isin(random_ids)]
        print(f"Random Sample index 0 {random_ids[0]}")

        krig = OrdinaryKriging(
            x=gpm25_krig["Easting"],
            y=gpm25_krig["Northing"],
            z=gpm25_krig["PM2.5"],
            variogram_model=variogram_model,
            # enable_statistics=True,
            nlags=nlags,
        )
        z, ss = krig.execute("grid", gridx, gridy)
        OK_pm25 = np.where(z < 0, 0, z)

        ds.assign_coords({"test": i})
        ds.assign_coords({"ids": np.arange(len(random_ids))})
        ds["pm25"] = (("y", "x"), OK_pm25)
        random_ids_list.append(random_ids.astype(str))

        list_ds.append(ds)
        print(f"Loop {i} time {datetime.now() - loopTime}")

    final_ds = xr.concat(list_ds, dim="test")
    final_ds["random_sample"] = (("test", "ids"), np.stack(random_ids_list))
    final_ds["random_sample"] = final_ds["random_sample"].astype(str)

    final_ds, encoding = compressor(final_ds)
    final_ds.to_netcdf(
        str(data_dir)
        + f"/OK-{krig.variogram_model.title()}-{nlags}-{int(frac*100)}.nc",
        encoding=encoding,
        mode="w",
    )


# fig = plt.figure(figsize=(8, 4))
# ax = fig.add_subplot(111)
# ax.plot(krig.lags, krig.semivariance, "go")
# ax.plot(
#     krig.lags,
#     krig.variogram_function(krig.variogram_model_parameters, krig.lags),
#     "k-",
# )
# ax.grid(True, linestyle="--", zorder=1, lw=0.5)
# fig_title = f"Coordinates type: {(krig.coordinates_type).title()}" + "\n"
# if krig.variogram_model == "linear":
#     fig_title += "Using '%s' Variogram Model" % "linear" + "\n"
#     fig_title += f"Slope: {krig.variogram_model_parameters[0]}" + "\n"
#     fig_title += f"Nugget: {krig.variogram_model_parameters[1]}"
# elif krig.variogram_model == "power":
#     fig_title += "Using '%s' Variogram Model" % "power" + "\n"
#     fig_title += f"Scale:  {krig.variogram_model_parameters[0]}" + "\n"
#     fig_title += f"Exponent: + {krig.variogram_model_parameters[1]}" + "\n"
#     fig_title += f"Nugget: {krig.variogram_model_parameters[2]}"
# elif krig.variogram_model == "custom":
#     print("Using Custom Variogram Model")
# else:
#     fig_title += f"Using {(krig.variogram_model).title()} Variogram Model" + "\n"
#     fig_title += (
#         f"Partial Sill: {np.round(krig.variogram_model_parameters[0])}" + "\n"
#     )
#     fig_title += (
#         f"Full Sill: {np.round(krig.variogram_model_parameters[0] + krig.variogram_model_parameters[2])}"
#         + "\n"
#     )
#     fig_title += f"Range: {np.round(krig.variogram_model_parameters[1])}" + "\n"
#     fig_title += f"Nugget: {np.round(krig.variogram_model_parameters[2],2)}"
# ax.set_title(fig_title, loc="left", fontsize=14)
# fig_title2 = (
#     f"Q1 = {np.round(krig.Q1,4)}"
#     + "\n"
#     + f"Q2 = {np.round(krig.Q2,4)}"
#     + "\n"
#     + f"cR = {np.round(krig.cR,4)}"
# )
# ax.set_title(fig_title2, loc="right", fontsize=14)

# ax.set_xlabel("Lag", fontsize=12)
# ax.set_ylabel("Semivariance", fontsize=12)
# ax.tick_params(axis="both", which="major", labelsize=12)
# plt.savefig(
#     str(img_dir)
#     + f"/ordinary-kriging-variogram-{krig.variogram_model}-{nlags}-{int(frac*100)}.png",
#     dpi=300,
#     bbox_inches="tight",
# )
# plt.close()
# print("------------------------")
