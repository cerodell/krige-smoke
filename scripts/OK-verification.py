import context
import numpy as np
import pandas as pd
import xarray as xr

# import osmnx as ox
# from skgstat import Variogram

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
# wesn = [-160.0,-52.0,32.,70.0] ## BSC Domain
wesn = [-129.0, -90.0, 40.0, 60.0]  ## Big Test Domain
# wesn = [-122.2, -105.5, 49.0, 56.5]  ## Test Domain
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


fig = px.scatter_mapbox(
    df_pm25,
    lat="lat",
    lon="lon",
    color="PM2.5",
    size="PM2.5",
    color_continuous_scale="RdYlGn_r",
    # hover_name="id",
    center={"lat": 52.722, "lon": -103.915},
    hover_data=["PM2.5"],
    mapbox_style="carto-positron",
    zoom=1.8,
)
fig.update_layout(margin=dict(l=0, r=100, t=30, b=10))
fig.show()


gpm25 = gpd.GeoDataFrame(
    df_pm25,
    crs="EPSG:4326",
    geometry=gpd.points_from_xy(df_pm25["lon"], df_pm25["lat"]),
).to_crs("EPSG:3347")
gpm25["Easting"], gpm25["Northing"] = gpm25.geometry.x, gpm25.geometry.y
gpm25.head()
gpm25.geometry.to_file("dataframe.shp")

# %%

# from shapely.geometry import MultiPoint
# import shapely.wkt

gpm25_poly = gpd.read_file(str(data_dir) + "/obs/outer_bounds")
gpm25_poly_buff = gpm25_poly.buffer(-80_000)
gpm25_buff = gpd.GeoDataFrame(
    {"geometry": gpd.GeoSeries(gpm25_poly_buff)}, crs=gpm25.crs
)

# test = MultiPoint(gpm25.geometry.values)
# test_poly = shapely.wkt.loads(test.convex_hull.wkt)


# gpm25_poly = gpd.GeoDataFrame({'geometry': gpd.GeoSeries(test_poly)}, crs=gpm25.crs, geometry='geometry')

# gpm25_poly = gpd.GeoDataFrame({'geometry': gpd.GeoSeries(gpm25.geometry.unary_union.convex_hull)}, crs=gpm25.crs)
# gpm25_poly_buff = gpm25_poly.buffer(-100_000)
gpm25_buff = gpd.GeoDataFrame(
    {"geometry": gpd.GeoSeries(gpm25_poly_buff)}, crs=gpm25.crs
)

gpm25_verif = gpd.sjoin(gpm25, gpm25_buff, predicate="within")

len(gpm25_verif)
random_sample = gpm25_verif.sample(frac=0.10)

gpm25_krig = gpm25[~gpm25.id.isin(random_sample.id)]


# %%


# gridx = np.arange(gpm25_krig.bounds.minx.min(), gpm25_krig.bounds.maxx.max(), resolution)
# gridy = np.arange(gpm25_krig.bounds.miny.min(), gpm25_krig.bounds.maxy.max(), resolution)

i = 16
krig = OrdinaryKriging(
    x=gpm25_krig["Easting"],
    y=gpm25_krig["Northing"],
    z=gpm25_krig["PM2.5"],
    variogram_model="gaussian",
    # enable_plotting=True,
    enable_statistics=True,
    # verbose=True,
    nlags=i,
)

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
ax.plot(krig.lags, krig.semivariance, "go")
ax.plot(
    krig.lags,
    krig.variogram_function(krig.variogram_model_parameters, krig.lags),
    "k-",
)
ax.grid(True, linestyle="--", zorder=1, lw=0.5)
fig_title = "Coordinates type: '%s'" % krig.coordinates_type + "\n"
if krig.variogram_model == "linear":
    fig_title += "Using '%s' Variogram Model" % "linear" + "\n"
    fig_title += f"Slope: {krig.variogram_model_parameters[0]}" + "\n"
    fig_title += f"Nugget: {krig.variogram_model_parameters[1]}"
elif krig.variogram_model == "power":
    fig_title += "Using '%s' Variogram Model" % "power" + "\n"
    fig_title += f"Scale:  {krig.variogram_model_parameters[0]}" + "\n"
    fig_title += f"Exponent: + {krig.variogram_model_parameters[1]}" + "\n"
    fig_title += f"Nugget: {krig.variogram_model_parameters[2]}"
elif krig.variogram_model == "custom":
    print("Using Custom Variogram Model")
else:
    fig_title += "Using '%s' Variogram Model" % krig.variogram_model + "\n"
    fig_title += f"Partial Sill: {krig.variogram_model_parameters[0]}" + "\n"
    fig_title += (
        f"Full Sill: {krig.variogram_model_parameters[0] + krig.variogram_model_parameters[2]}"
        + "\n"
    )
    fig_title += f"Range: {krig.variogram_model_parameters[1]}" + "\n"
    fig_title += f"Nugget: {krig.variogram_model_parameters[2]}"
ax.set_title(fig_title, loc="left", fontsize=14)
ax.set_xlabel("Lag", fontsize=12)
ax.set_ylabel("Semivariance", fontsize=12)
ax.tick_params(axis="both", which="major", labelsize=12)
plt.savefig(
    str(img_dir) + f"/ordinary-kriging-variogram-gaussian{i}.png",
    dpi=300,
    bbox_inches="tight",
)
print("------------------------")
# print(f"nlags of {6}")
print("Q1 =", krig.Q1)
print("Q2 =", krig.Q2)
print("cR =", krig.cR, "\n")

# print("------------------------")


# z, ss = krig.execute("grid", gridx, gridy)
# OK_pm25 = np.where(z < 0, 0, z)


# print(krig.variogram_model_parameters[1]/1000)

# polygons, values = pixel2poly(gridx, gridy, OK_pm25, resolution)

# pm25_model = (gpd.GeoDataFrame({"PM_25_modelled": values}, geometry=polygons, crs="EPSG:3347")
#                  .to_crs("EPSG:4326")
#                  )

# # fig = px.choropleth_mapbox(pm25_model, geojson=pm25_model.geometry, locations=pm25_model.index,
# #                            color="PM_25_modelled", color_continuous_scale="RdYlGn_r",
# #                            center={"lat": 52.261, "lon": -123.246}, zoom=3.5,
# #                            mapbox_style="carto-positron")
# # fig.update_layout(margin=dict(l=0, r=0, t=30, b=10))
# # fig.update_traces(marker_line_width=0)
# # fig.show()

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
