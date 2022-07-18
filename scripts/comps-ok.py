# %% [markdown]
# # Ordinary Kriging

# %%
import context
import numpy as np
import pandas as pd
import salem

import geopandas as gpd
import plotly.express as px

from pykrige.ok import OrdinaryKriging


from utils.utils import pixel2poly

from context import data_dir, img_dir
import time


# %%
df = pd.read_csv(str(data_dir) + "/obs/gpm25.csv")
gpm25 = gpd.GeoDataFrame(
    df,
    crs="EPSG:4326",
    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
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
    center={"lat": 49.5, "lon": -107.0},
    zoom=2.5,
    mapbox_style="carto-positron",
)
fig.update_layout(margin=dict(l=0, r=0, t=30, b=10))
fig.update_traces(marker_line_width=0)
