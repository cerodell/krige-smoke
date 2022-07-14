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
frac = 0.10

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

# %%

gpm25_poly = gpd.read_file(str(data_dir) + "/obs/outer_bounds")
gpm25_poly_buff = gpm25_poly.buffer(-80_000)
gpm25_buff = gpd.GeoDataFrame(
    {"geometry": gpd.GeoSeries(gpm25_poly_buff)}, crs=gpm25.crs
)
gpm25_verif = gpd.sjoin(gpm25, gpm25_buff, predicate="within")


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

list_ds = []

for i in range(0, 20):
    loopTime = datetime.now()

    print(i)
    ds = grid_ds
    random_sample = gpm25_verif.sample(frac=frac)
    gpm25_krig = gpm25[~gpm25.id.isin(random_sample.id)]

    krig = OrdinaryKriging(
        x=gpm25_krig["Easting"],
        y=gpm25_krig["Northing"],
        z=gpm25_krig["PM2.5"],
        variogram_model=variogram_model,
        enable_statistics=True,
        nlags=nlags,
    )
    z, ss = krig.execute("grid", gridx, gridy)
    OK_pm25 = np.where(z < 0, 0, z)

    ds.assign_coords({"test": i})
    ds.assign_coords({"ids": np.arange(len(random_sample.id.values))})
    ds["pm25"] = (("y", "x"), OK_pm25)
    ds["random_sample"] = ("ids", random_sample.id.values.astype(str))
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

    # gridxx, gridyy = np.meshgrid(gridx, gridy)

    # ds = xr.Dataset(
    #     data_vars={
    #         "OK_pm25": (["x", "y"], OK_pm25.astype("float32")),
    #         "random_sample": ("ids", random_sample.id.values.astype(str)),
    #     },
    #     coords={
    #         "gridx": (["x", "y"], gridxx),
    #         "gridy": (["x", "y"], gridyy),
    #         "test": i,
    #         "ids": np.arange(len(random_sample.id.values)),
    #     },
    #     attrs=dict(description=krig.variogram_model.title()),
    # )
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
    str(data_dir) + f"/OK-{krig.variogram_model.title()}-{nlags}-{int(frac*100)}.nc",
    encoding=encoding,
    mode="w",
)


# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# na = world.query('continent == "North America"')
# minx, miny, maxx, maxy = pm25_model.geometry.total_bounds

# fig, ax = plt.subplots(1, 1)

# na.boundary.plot(ax = ax)
# pm25_model.plot("PM_25_modelled", cmap="RdYlGn_r", ax=ax, zorder = 10)
# ax.set_xlim(minx - .1, maxx + .1)
# ax.set_ylim(miny - .1, maxy + .1)
# print("--- %s seconds ---" % (time.time() - start_time))
# # %%

# PM2_5 = random_sample['PM2.5'].values
# Easting = random_sample.Easting.values
# Northing = random_sample.Northing.values
# # ii = np.where(gridx==Easting)
# # jj = np.where(gridy==Northing[0])

# gridxx, gridyy = np.meshgrid(gridx, gridy)
# shape = gridxx.shape
# ## create dataframe with columns of all lat/long in the domian...rows are cord pairs
# locs = pd.DataFrame({"gridx": gridxx.ravel(), "gridy": gridyy.ravel()})
# ## build kdtree
# tree = KDTree(locs)

# south_north,  west_east, ids = [], [], []
# for loc in random_sample.itertuples(index=True, name='Pandas'):
#     ## arange wx station lat and long in a formate to query the kdtree
#     single_loc = np.array([loc.Easting, loc.Northing]).reshape(1, -1)

#     ## query the kdtree retuning the distacne of nearest neighbor and the index on the raveled grid
#     dist, ind = tree.query(single_loc, k=1)
#     print(dist)
#     ## if condition passed reformate 1D index to 2D indexes
#     ind_2D = np.unravel_index(int(ind), shape)
#     ## append the indexes to lists
#     ids.append(loc.id)
#     south_north.append(ind_2D[0])
#     west_east.append(ind_2D[1])

# OK_pm25_points = OK_pm25[south_north, west_east]
# random_sample['modeled_PM2.5'] = OK_pm25_points

# random_sample['modeled_obs'] = random_sample['modeled_PM2.5'].values - random_sample['PM2.5'].values
# print(random_sample['modeled_PM2.5'].corr(random_sample['PM2.5']))
# fig = px.scatter_mapbox(random_sample,
#                     lat='lat',
#                     lon='lon',
#                     color='modeled_obs',
#                     # size='modeled_obs',
#                     color_continuous_scale="RdYlGn_r",
#                     # hover_name="id",
#                     center={"lat": 52.722, "lon": -103.915},
#                     hover_data= ['modeled_obs'],
#                     mapbox_style="carto-positron",
#                     zoom=1.8,
#                     )
# fig.update_layout(margin=dict(l=0, r=100, t=30, b=10))
# fig.show()
# plt.scatter(OK_pm25_points,random_sample['PM2.5'])
