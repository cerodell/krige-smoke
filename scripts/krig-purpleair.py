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


variogram_models = [
    "linear",
    "power",
    "gaussian",
    "spherical",
    "exponential",
    "hole-effect",
]


# # Before droping outliers
# fig = plt.figure()
# xr.plot.hist(ds['PM2.5'])

# After droping outliers
ds = ds.where(ds["PM2.5"] < 1000, drop=True)
ds = ds.where(ds["PM2.5"] > 0, drop=True)
mean = ds["PM2.5"].mean()
sd = ds["PM2.5"].std()
sd_ds = ds.where(
    (ds["PM2.5"] > mean - 2 * sd) & (ds["PM2.5"] < mean + 2 * sd), drop=True
)
# fig = plt.figure()
# xr.plot.hist(sd_ds['PM2.5'])
# fig.show()


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

print(len(df_pm25))
## Interpolate pm25 after removing outliers
# lats = df_pm25["lat"].values
# lons = df_pm25["lon"].values
# pm25 = df_pm25["PM2.5"].values


# %%
gpm25 = gpd.GeoDataFrame(
    df_pm25,
    crs="EPSG:4326",
    geometry=gpd.points_from_xy(df_pm25["lon"], df_pm25["lat"]),
).to_crs("EPSG:3347")
gpm25["Easting"], gpm25["Northing"] = gpm25.geometry.x, gpm25.geometry.y
gpm25.head()

gridx = np.arange(gpm25.bounds.minx.min(), gpm25.bounds.maxx.max(), resolution)
gridy = np.arange(gpm25.bounds.miny.min(), gpm25.bounds.maxy.max(), resolution)
for i in range(1, 30):
    plt.close()
    krig_start_time = time.time()
    krig = OrdinaryKriging(
        x=gpm25["Easting"],
        y=gpm25["Northing"],
        z=gpm25["PM2.5"],
        variogram_model="spherical",
        # enable_plotting=True,
        enable_statistics=True,
        # verbose=True,
        nlags=i,
    )
    print(f"nlags of {i}")
    print("Q1 =", krig.Q1)
    print("Q2 =", krig.Q2)
    print("cR =", krig.cR, "\n")
    print("--- %s seconds ---" % (time.time() - krig_start_time))
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
        str(img_dir) + f"/ordinary-kriging-variogram-spherical{i}.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("------------------------")


# krig = OrdinaryKriging(
#     x=gpm25["Easting"],
#     y=gpm25["Northing"],
#     z=gpm25["PM2.5"],
#     variogram_model="spherical",
#     # enable_plotting=True,
#     enable_statistics=True,
#     # verbose=True,
#     nlags=6,
# )
# print(f"nlags of {6}")
# print("Q1 =", krig.Q1)
# print("Q2 =", krig.Q2)
# print("cR =", krig.cR, "\n")

# print("------------------------")


# z, ss = krig.execute("grid", gridx, gridy)
# OK_pm25 = np.where(z < 0, 0, z)

# %%
# fig = plt.figure(figsize=(8, 4))
# ax = fig.add_subplot(111)
# ax.plot(krig.lags, krig.semivariance, "go")
# ax.plot(
#     krig.lags,
#     krig.variogram_function(krig.variogram_model_parameters, krig.lags),
#     "k-",
# )
# ax.grid(True, linestyle="--", zorder=1, lw=0.5)
# fig_title = "Coordinates type: '%s'" % krig.coordinates_type + "\n"
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
#     fig_title += "Using '%s' Variogram Model" % krig.variogram_model + "\n"
#     fig_title += f"Partial Sill: {krig.variogram_model_parameters[0]}" + "\n"
#     fig_title += (
#         f"Full Sill: {krig.variogram_model_parameters[0] + krig.variogram_model_parameters[2]}"
#         + "\n"
#     )
#     fig_title += f"Range: {krig.variogram_model_parameters[1]}" + "\n"
#     fig_title += f"Nugget: {krig.variogram_model_parameters[2]}"
# ax.set_title(fig_title, loc="left", fontsize=14)
# ax.set_xlabel("Lag", fontsize=12)
# ax.set_ylabel("Semivariance", fontsize=12)
# ax.tick_params(axis="both", which="major", labelsize=12)

# print(krig.variogram_model_parameters[1]/1000)

# polygons, values = pixel2poly(gridx, gridy, OK_pm25, resolution)

# pm25_model = (gpd.GeoDataFrame({"PM_25_modelled": values}, geometry=polygons, crs="EPSG:3347")
#                  .to_crs("EPSG:4326")
#                  )

# fig = px.choropleth_mapbox(pm25_model, geojson=pm25_model.geometry, locations=pm25_model.index,
#                            color="PM_25_modelled", color_continuous_scale="RdYlGn_r",
#                            center={"lat": 52.261, "lon": -123.246}, zoom=3.5,
#                            mapbox_style="carto-positron")
# fig.update_layout(margin=dict(l=0, r=0, t=30, b=10))
# fig.update_traces(marker_line_width=0)
# fig.show()

world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
na = world.query('continent == "North America"')
minx, miny, maxx, maxy = pm25_model.geometry.total_bounds

fig, ax = plt.subplots(1, 1)

na.boundary.plot(ax=ax)
pm25_model.plot("PM_25_modelled", cmap="RdYlGn_r", ax=ax, zorder=10)
ax.set_xlim(minx - 0.1, maxx + 0.1)
ax.set_ylim(miny - 0.1, maxy + 0.1)
print("--- %s seconds ---" % (time.time() - start_time))
# %%


# ## bring in state/prov boundaries
# states_provinces = cfeature.NaturalEarthFeature(
#     category="cultural",
#     name="admin_1_states_provinces_lines",
#     scale="50m",
#     facecolor="none",
# )


# cmap = mpl.cm.jet
# norm = matplotlib.colors.Normalize(vmin=np.min(pm25).astype(int), vmax=np.max(pm25).astype(int) + 1)
# levels = np.arange(np.min(pm25).astype(int),np.max(pm25).astype(int) + 1, 0.1)


# # %%
# for variogram_model in variogram_models:
#     plt.close()
#     OK = OrdinaryKriging(
#                             lons,
#                             lats,
#                             pm25,
#                             variogram_model=variogram_model,
#                             verbose=True,
#                             enable_plotting=True,
#                             nlags=20
#                         )
#     plt.savefig(str(img_dir) + f"/ordinary-kriging-variogram-{variogram_model}.png", dpi=300, bbox_inches="tight")
#     plt.close()
#     OK_pm25, ss1 = OK.execute('grid', grid_lon, grid_lat)
#     OK_pm25 = np.where(OK_pm25<0, 0, OK_pm25)

#     # create fig and axes using intended projection
#     fig = plt.figure(figsize=(10,6))
#     pm25_crs = ccrs.PlateCarree()
#     ax = fig.add_subplot(1, 1, 1, projection=pm25_crs)
#     CS = ax.contourf(grid_lon, grid_lat, OK_pm25, levels=levels,cmap=cmap, norm=norm)
#     ax.add_feature(states_provinces, linewidth=0.5, edgecolor="black", zorder=10)
#     ax.add_feature(cfeature.BORDERS, zorder=10,  lw = 0.7)
#     ax.add_feature(cfeature.COASTLINE, zorder=10, lw = 0.7)
#     gl = ax.gridlines(
#         crs=ccrs.PlateCarree(),
#         draw_labels=True,
#         linewidth=1,
#         color="gray",
#         alpha=0.5,
#         linestyle="--",
#         zorder=2,
#     )
#     gl.xlabels_top = False
#     gl.ylabels_right = False
#     gl.xlabel_style = {"size": 14}
#     gl.ylabel_style = {"size": 14}
#     CS = ax.scatter(lons,lats,c = pm25,norm=norm, cmap =cmap)
#     ax.set_title(f"Ordinary Kriging: {variogram_model.title()}", fontsize = 16)
#     cbar = plt.colorbar(CS, ax=ax, pad=0.004, location="right", shrink = 0.65)
#     fig.tight_layout()
#     # fig.show()
#     plt.savefig(str(img_dir) + f"/ordinary-kriging-{variogram_model}.png", dpi=300, bbox_inches="tight")
#     print(f"ordinary kriging {variogram_model} done")

# print('##############################################################')


# # %%
# for variogram_model in variogram_models:
#     plt.close()
#     UK = UniversalKriging(
#         lons,
#         lats,
#         pm25,
#         variogram_model=variogram_model,
#         drift_terms=["regional_linear"],
#         verbose=True,
#         # enable_plotting = True
#     )
#     UK_pm25, ss = UK.execute("grid", grid_lon, grid_lat)
#     UK_pm25 = np.where(UK_pm25<0, 0, UK_pm25)

#     # create fig and axes using intended projection
#     fig = plt.figure(figsize=(10,6))
#     pm25_crs = ccrs.PlateCarree()
#     ax = fig.add_subplot(1, 1, 1, projection=pm25_crs)
#     CS = ax.contourf(grid_lon, grid_lat, UK_pm25, levels=levels,cmap=cmap, norm=norm)
#     ax.add_feature(states_provinces, linewidth=0.5, edgecolor="black", zorder=10)
#     ax.add_feature(cfeature.BORDERS, zorder=10,  lw = 0.7)
#     ax.add_feature(cfeature.COASTLINE, zorder=10, lw = 0.7)
#     gl = ax.gridlines(
#         crs=ccrs.PlateCarree(),
#         draw_labels=True,
#         linewidth=1,
#         color="gray",
#         alpha=0.5,
#         linestyle="--",
#         zorder=2,
#     )
#     gl.xlabels_top = False
#     gl.ylabels_right = False
#     gl.xlabel_style = {"size": 14}
#     gl.ylabel_style = {"size": 14}
#     CS = ax.scatter(lons,lats,c = pm25,norm=norm, cmap =cmap)
#     # cbar = fig.colorbar(CS)
#     cbar = plt.colorbar(CS, ax=ax, pad=0.004, location="right", shrink = 0.65)
#     ax.set_title(f"Universal Kriging: {variogram_model.title()}", fontsize = 16)

#     fig.tight_layout()
#     # fig.show()
#     plt.savefig(str(img_dir) + f"/universal-kriging-{variogram_model}.png", dpi=300, bbox_inches="tight")
#     print(f"universal kriging {variogram_model} done")


# # %%
