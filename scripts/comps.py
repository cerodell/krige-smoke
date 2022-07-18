# %% [markdown]
# # Kriging

# %% [markdown]
# ## PurpleAir and Goverment Operatated Air quality Data
# - See find-pa-station.py,  get-pa-data.py and remormate-gov.py for how I obtanined a nd remotated the data for ease of use.


# %% [markdown]
# ## Case Study.

# I chose to fouce on July 2021, post heat dome with high fire activeity in souther BC and the PNW of the US.

# TODO add image of smoke from nasa worldview.

# %%
import context
import numpy as np
import pandas as pd
import xarray as xr
import salem


import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from shapely.geometry import Polygon
import shapely

import matplotlib.pyplot as plt
from utils.utils import pixel2poly

from context import data_dir, img_dir
import time

start_time = time.time()
# %% [markdown]
# ### Choose date and time of interest to test kriging
dot = "2021-07-16T22:00:00"
# %%
## Define domain of interest... this is the same bounds as the BlueSky Canada Forecasts
# wesn = [-160.0,-52.0,32.,70.0]
# wesn = [-129.0, -90.0, 40.0, 60.0]  ## Big Test Domain
wesn = [-126, -105.5, 45.0, 56.5]

# wesn = [-129.0, -90.0, 40.0, 60.0]  ## Big Test Domain

## Open Government AQ data and index on dot
gov_ds = xr.open_dataset(str(data_dir) + f"/gov_aq.nc")
gov_ds = gov_ds.sel(datetime="2021-07-16T22:00:00")

## Open PurpleAir AQ data, index on dot and drop variables to make ds concat with gov_ds
pa_ds = xr.open_dataset(str(data_dir) + f"/purpleair_north_america.nc")
pa_ds = pa_ds.sel(datetime="2021-07-16T22:00:00")
pa_ds = pa_ds.drop(["PM1.0", "PM10.0", "pressure", "PM2.5_ATM"])

## concat both ds on as station id
ds = xr.concat([pa_ds, gov_ds], dim="id")

# Drop outliers by..
ds = ds.where(ds["PM2.5"] < 1000, drop=True)  ## Erroneously high values
ds = ds.where(ds["PM2.5"] > 0, drop=True)  ## Non-physical negative values
mean = ds["PM2.5"].mean()  ## outside two standard deviation
sd = ds["PM2.5"].std()
sd_ds = ds.where(
    (ds["PM2.5"] > mean - 2 * sd) & (ds["PM2.5"] < mean + 2 * sd), drop=True
)

sd_ds

# Convert our dataset to a dataframe and drop all aq stations outside our domain
df_pm25 = sd_ds["PM2.5"].to_dataframe().reset_index()
df_pm25 = df_pm25.loc[
    (df_pm25["lat"] > wesn[2])
    & (df_pm25["lat"] < wesn[3])
    & (df_pm25["lon"] > wesn[0])
    & (df_pm25["lon"] < wesn[1])
]

df_pm25.head()
# %% [markdown]
# ### Plot Data
# Lets look at the data by first plotting the distribution of the measured PM 2.5 measured values.
# %%
fig = ff.create_distplot([sd_ds["PM2.5"].values], ["PM2.5"], colors=["green"])
fig.show()

# %% [markdown]
# Now lets spatially look at the data by a scatter plot of the measured PM 2.5 values at each station.
# %%
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

# %% [markdown]
# We can see how the fires in BC are creating poor air quality in the east rockies and praires/plaines.


# %% [markdown]
# ### Reproject Data

# We want to convert the data to the linear, meter-based Lambert projection (EPSG:3347) recommended by Statistics Canada. This is helpful as lat/lon coordinates are not good for measuring distances which is important for interpolating data.

# %%

gpm25 = gpd.GeoDataFrame(
    df_pm25,
    crs="EPSG:4326",
    geometry=gpd.points_from_xy(df_pm25["lon"], df_pm25["lat"]),
).to_crs("EPSG:3347")
gpm25["Easting"], gpm25["Northing"] = gpm25.geometry.x, gpm25.geometry.y
gpm25.head()


# %% [markdown]
# ### Create Grid
# Here we will create a grid we want to use for the interpolate on.

# %%
resolution = 20_000  # cell size in meters

gridx = np.arange(gpm25.bounds.minx.min(), gpm25.bounds.maxx.max(), resolution)
gridy = np.arange(gpm25.bounds.miny.min(), gpm25.bounds.maxy.max(), resolution)


krig_ds = salem.Grid(
    nxny=(len(gridx), len(gridy)),
    dxdy=(resolution, resolution),
    x0y0=(gpm25.bounds.minx.min(), gpm25.bounds.miny.min()),
    proj="epsg:3347",
    pixel_ref="corner",
).to_dataset()

krig_ds

# %% [markdown]
# ## Krig
# ### Ordinary Kriging
# %%
nlags = 15
variogram_model = "spherical"

krig = OrdinaryKriging(
    x=gpm25["Easting"],
    y=gpm25["Northing"],
    z=gpm25["PM2.5"],
    variogram_model=variogram_model,
    nlags=nlags,
)
z, ss = krig.execute("grid", gridx, gridy)
OK_pm25 = np.where(z < 0, 0, z)

# krig_ds["OK_pm25"] = (("y", "x"), OK_pm25)

polygons, values = pixel2poly(gridx, gridy, OK_pm25, resolution)
pm25_model = gpd.GeoDataFrame(
    {"PM_25_modelled": values}, geometry=polygons, crs="EPSG:3347"
).to_crs("EPSG:4326")

fig = px.choropleth_mapbox(
    pm25_model,
    geojson=pm25_model.geometry,
    locations=pm25_model.index,
    color="PM_25_modelled",
    color_continuous_scale="RdYlGn_r",
    center={"lat": 52.261, "lon": -123.246},
    zoom=3.5,
    mapbox_style="carto-positron",
)
fig.update_layout(margin=dict(l=0, r=0, t=30, b=10))
fig.update_traces(marker_line_width=0)


# %% [markdown]
# ### Universal Kriging
