# %% [markdown]
# # Case Study
#
# As mentioned before, we will test kringing using observation of poor air quality due to wildfire smoke. We chose July 16, 2021, as there was a high concentration of smoke across a large portion of North America.
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
# Satellite observations of wildfire smoke in the visible spectrum on July 16, 2021.
# - Also shown are the satellite hotspot detects (ie wildfire) in orange.
# %%
img = mpimg.imread(str(data_dir) + "/obs/worldview.jpeg")
fig = px.imshow(img)
fig.update_layout(margin=dict(l=10, r=10, t=30, b=30))
fig.show()


# %% [markdown]
# ## Data Setup

# %% [markdown]
# ### Air quality monitors
# - See [find-pa-station.py](https://github.com/cerodell/krige-smoke/blob/main/scripts/find-pa-sations.py),
#  [get-pa-data.py](https://github.com/cerodell/krige-smoke/blob/main/scripts/get-pa-data.py)
#  and [reformate-gov.py](https://github.com/cerodell/krige-smoke/blob/main/scripts/reformate-gov.py) for how I obtained and combined data in a single dataset.


# %% [markdown]
# ### Datetime and domain
# Here we define a datetime and domain of interest to test kriging.

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
# ### Clean Data
# Remove outliers by:
#
# - Erroneously high values
#
# - Non-physical negative values
#
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
# - Convert our dataset to a dataframe and drop all aq stations outside our domain of interest
# %%
df_pm25 = sd_ds["PM2.5"].to_dataframe().reset_index()
df_pm25 = df_pm25.loc[
    (df_pm25["lat"] > wesn[2])
    & (df_pm25["lat"] < wesn[3])
    & (df_pm25["lon"] > wesn[0])
    & (df_pm25["lon"] < wesn[1])
]

print(f"Number of AQ Monitors: {len(df_pm25)}")
df_pm25.head()
# %% [markdown]
# ### Plot Data
# #### Distribution
# Look at the data by plotting the measured PM 2.5 values distribution.
# %%
fig = ff.create_distplot([df_pm25["PM2.5"].values], ["PM2.5"], colors=["green"])
fig.show()

# %% [markdown]
# The data has

# %% [markdown]
# #### Spatial scatter plot
# Now,  look at the data by a scatter plot of the measured PM 2.5 values at each station.
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
# We can see how the fires are creating poor air quality in the eastern Rockies and prairies/plains.


# %% [markdown]
# ### Reproject Data

# We want to convert the data to the linear, meter-based Lambert projection (EPSG:3347) recommended by Statistics Canada. This conversion is helpful as lat/lon coordinates are not as suitable for measuring distances which is vital for interpolating data.

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
