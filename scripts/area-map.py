import context
import numpy as np
import pandas as pd
import xarray as xr

import plotly.express as px

import geopandas as gpd
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
# gpm25.geometry.to_file(str(data_dir)+"/obs/dataframe.shp")

# %%


gpm25_poly = gpd.read_file(str(data_dir) + "/obs/outer_bounds")
gpm25_poly_buff = gpm25_poly.buffer(-80_000)
gpm25_buff = gpd.GeoDataFrame(
    {"geometry": gpd.GeoSeries(gpm25_poly_buff)}, crs=gpm25.crs
)

gpm25_buff = gpm25_buff.to_crs("EPSG:4326")
gpm25_buff["layer"] = ["verification"]
gpm25_poly = gpm25_poly.to_crs("EPSG:4326")
gpm25_poly["layer"] = ["study_area"]
df_pm25["geometry"] = np.nan
gpm25_poly = pd.concat([gpm25_poly, gpm25_buff, df_pm25], ignore_index=True)
# gpm25_poly['id'] = [0,1]


# fig = px.choropleth_mapbox(gpm25_poly, geojson=gpm25_poly.geometry.loc[:1],
#                           #  locations=gpm25_poly.index.loc[:1],
#                           center={"lat": 52.722, "lon": -103.915},
#                             zoom=1.8,
#                             mapbox_style="carto-positron",
#                             color='id',

# )
# fig.add_scattermapbox(
#     # df_pm25,
#     lat=gpm25_poly['lat'].loc[2:],
#     lon=gpm25_poly['lon'].loc[2:],
#     color=gpm25_poly['PM2.5'].loc[2:],
#     size=gpm25_poly['PM2.5'].loc[2:],
#     color_continuous_scale="RdYlGn_r",
#     # hover_name="id",
#     # center={"lat": 52.722, "lon": -103.915},
#     hover_data=[gpm25_poly['PM2.5'].loc[2:]],
#     # mapbox_style="carto-positron",
#     # cliponaxis = True,
#     # zoom=1.8,
# )
# fig.show()
