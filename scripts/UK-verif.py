import context
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import KDTree
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
from pylab import *

from context import data_dir, img_dir


nlags = 15
frac = 0.1
variogram_model = "spherical"


gpm25 = pd.read_csv(str(data_dir) + "/obs/gpm25.csv")

UK_ds = xr.open_dataset(
    str(data_dir) + f"/UK-dir-{variogram_model.title()}-{nlags}-{int(frac*100)}.nc"
)


fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
colors = []
cmap = cm.get_cmap("seismic", len(UK_ds.test) + 1)  # PiYG
for i in range(cmap.N):
    rgba = cmap(i)
    colors.append(matplotlib.colors.rgb2hex(rgba))

R_2 = []
modeled, observed = [], []

for i in range(len(UK_ds.test)):
    print(i)
    UK = UK_ds.isel(test=i)
    UK_pm25 = UK.pm25.values
    print(UK.random_sample.values[:2])
    random_sample = gpm25[gpm25.id.isin(UK.random_sample.values)].copy()
    y = xr.DataArray(
        np.array(random_sample["Northing"]),
        dims="ids",
        coords=dict(ids=random_sample.id.values),
    )
    x = xr.DataArray(
        np.array(random_sample["Easting"]),
        dims="ids",
        coords=dict(ids=random_sample.id.values),
    )
    pm25_points = UK.pm25.interp(x=x, y=y, method="linear")

    random_sample["modeled_PM2.5"] = pm25_points
    modeled.append(pm25_points.values)
    observed.append(random_sample["PM2.5"].values)
    ax.scatter(pm25_points, random_sample["PM2.5"], color=colors[i])
    r = random_sample["modeled_PM2.5"].corr(random_sample["PM2.5"])
    R_2.append(r)
    print(r)
plt.close()
modeled, observed = np.ravel(modeled), np.ravel(observed)

print(
    f"root mean squared error {mean_squared_error(observed, modeled, squared = False)}"
)
print(f"mean absolute error {mean_absolute_error(observed, modeled)}")
print(f"Average R^2 {np.mean(R_2)}")


df = pd.DataFrame({"modeled_pm25": modeled, "observed_pm25": observed})
