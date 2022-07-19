# %% [markdown]
# # Data Setup

# %% [markdown]
# ### PurpleAir and Government operated air quality monitors
# - See find-pa-station.py,  get-pa-data.py and remormate-gov.py for how I obtained and combined data in a single dataset.


# %% [markdown]
# Load python modules
# %%
import context
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

import plotly.express as px
import matplotlib.image as mpimg
import plotly.figure_factory as ff

from context import data_dir

# %% [markdown]
# ### Case Study.

# I chose to focus on July 2021, post heat dome with high fire activity in souther BC and the PNW of the US.

# %%

img = mpimg.imread(str(data_dir) + "/obs/worldview.jpeg")
fig = px.imshow(img)
fig.update_layout(margin=dict(l=10, r=10, t=30, b=30))
fig.show()

# %% [markdown]
# ### Choose date and time of interest to test kriging
# %%
## Define domain and datetime of interest
wesn = [-129.0, -90.0, 40.0, 60.0]
dot = "2021-07-16T22:00:00"


## Open Government AQ data and index on dot
gov_ds = xr.open_dataset(str(data_dir) + f"/gov_aq.nc")
gov_ds = gov_ds.sel(datetime=dot)

## Open PurpleAir AQ data, index on dot and drop variables to make ds concat with gov_ds
pa_ds = xr.open_dataset(str(data_dir) + f"/purpleair_north_america.nc")
pa_ds = pa_ds.sel(datetime=dot)
pa_ds = pa_ds.drop(["PM1.0", "PM10.0", "pressure", "PM2.5_ATM"])

## concat both datasets on as station id
ds = xr.concat([pa_ds, gov_ds], dim="id")

# %% [markdown]
# ### Remove outliers by..
# - Erroneously high values
# - Non-physical negative values
# - Outside two standard deviation
# %%
ds = ds.where(ds["PM2.5"] < 1000, drop=True)  ## Erroneously high values
ds = ds.where(ds["PM2.5"] > 0, drop=True)  ## Non-physical negative values
mean = ds["PM2.5"].mean()  ## outside two standard deviation
sd = ds["PM2.5"].std()
sd_ds = ds.where(
    (ds["PM2.5"] > mean - 2 * sd) & (ds["PM2.5"] < mean + 2 * sd), drop=True
)

sd_ds


# %% [markdown]
# ### Reformat Date
# - Convert our dataset to a dataframe and drop all aq stations outside our domain
# %%
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
# #### Distribution
# - Lets look at the data by first plotting the distribution of the measured PM 2.5 measured values.
# %%
fig = ff.create_distplot([sd_ds["PM2.5"].values], ["PM2.5"], colors=["green"])
fig.show()

# %% [markdown]
# #### Spatial scatter plot
# Now lets spatially look at the data by a scatter plot of the measured PM 2.5 values at each station.
# %%
fig = px.scatter_mapbox(
    df_pm25,
    lat="lat",
    lon="lon",
    color="PM2.5",
    size="PM2.5",
    color_continuous_scale="jet",
    # hover_name="id",
    center={"lat": 50.0, "lon": -110.0},
    hover_data=["PM2.5"],
    mapbox_style="carto-positron",
    zoom=3.0,
)
fig.update_layout(margin=dict(l=0, r=100, t=30, b=10))
fig.show()

# %% [markdown]
# We can see how the fires in BC are creating poor air quality in the east rockies and prairies/plains.


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
# ### Onto Ordinary Kriging...
