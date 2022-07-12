import context
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import KDTree

import matplotlib.pyplot as plt
from pylab import *

from context import data_dir, img_dir


# nlags = 16
# variogram_model = 'gaussian'
nlags = 15
frac = 0.5
variogram_model = "spherical"


gpm25 = pd.read_csv(str(data_dir) + "/obs/gpm25.csv")


OK_ds = xr.open_dataset(
    str(data_dir) + f"/{variogram_model.title()}-{nlags}-{int(frac*100)}.nc"
)

gridx, gridy = OK_ds.gridx.values, OK_ds.gridy.values
shape = gridx.shape

## create dataframe with columns of all cord pairs
locs = pd.DataFrame({"gridx": gridx.ravel(), "gridy": gridy.ravel()})
## build kdtree
tree = KDTree(locs)

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
colors = []
cmap = cm.get_cmap("seismic", len(OK_ds.test) + 1)  # PiYG
for i in range(cmap.N):
    rgba = cmap(i)
    colors.append(matplotlib.colors.rgb2hex(rgba))

R_2 = []
for i in range(len(OK_ds.test)):

    OK = OK_ds.isel(test=i)
    OK_pm25 = OK.OK_pm25.values
    random_sample = gpm25[gpm25.id.isin(OK.random_sample.values)].copy()
    # print(OK.random_sample.values)
    south_north, west_east, ids = [], [], []
    for loc in random_sample.itertuples(index=True, name="Pandas"):
        ## arange wx station lat and long in a formate to query the kdtree
        single_loc = np.array([loc.Easting, loc.Northing]).reshape(1, -1)

        ## query the kdtree retuning the distacne of nearest neighbor and the index on the raveled grid
        dist, ind = tree.query(single_loc, k=1)
        # print(dist)
        ## if condition passed reformate 1D index to 2D indexes
        ind_2D = np.unravel_index(int(ind), shape)
        ## append the indexes to lists
        ids.append(loc.id)
        south_north.append(ind_2D[0])
        west_east.append(ind_2D[1])

    OK_pm25_points = OK_pm25[south_north, west_east]
    random_sample["modeled_PM2.5"] = OK_pm25_points
    ax.scatter(OK_pm25_points, random_sample["PM2.5"], color=colors[i])
    r = random_sample["modeled_PM2.5"].corr(random_sample["PM2.5"])
    R_2.append(r)
    print(r)

print(f"Average R^2 {np.mean(R_2)}")
