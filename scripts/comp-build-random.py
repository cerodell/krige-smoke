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


list_ds, random_ids_list = [], []
for i in range(0, 10):
    loopTime = datetime.now()
    gpm25_veriff = gpm25_verif.sample(frac=1).reset_index(drop=True)
    random_sample = gpm25_veriff.sample(frac=frac, replace=True, random_state=1)
    random_ids = random_sample.id.values
    # print(random_ids)
    gpm25_krig = gpm25.loc[~gpm25.id.isin(random_sample.id)]
    print(f"Random Sample index 0 {random_ids[0]}")

    random_ids_list.append(random_ids.astype(str))

# np.savetxt(str(data_dir) + '/rand-samples.txt', np.array(random_ids_list), delimiter=',')

# Displaying the array
save_array = np.array(random_ids_list)
test_array = np.arange(0, 10)
dict(zip(test_array, random_ids_list))
dataset = pd.DataFrame(dict(zip(test_array, random_ids_list)))
dataset.to_csv(str(data_dir) + f"/random-samples-{int(frac*100)}.csv")

# print('Array:\n', save_array)
# file = open(str(data_dir) + '/rand-samples.txt', "w+")

# # Saving the 2D array in a text file
# content = str(save_array)
# file.write(content)
# file.close()


# # Displaying the contents of the text file
# file = open(str(data_dir) + '/rand-samples.txt', "r")
# content = file.read()

# print("\nContent in file2.txt:\n", content)
# file.close()
