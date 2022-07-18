import context
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import seaborn as sns
from context import data_dir, img_dir
from matplotlib import pyplot as plt


gpm25 = pd.read_csv(str(data_dir) + "/obs/gpm25.csv")
pathlistUK = sorted(Path(str(data_dir)).glob(f"UK-*"))
pathlistOK = sorted(Path(str(data_dir)).glob(f"OK-*"))

pathlist = pathlistOK + pathlistUK


def buildsats(path):
    UK_ds = xr.open_dataset(str(path))
    modeled, observed = [], []
    for i in range(len(UK_ds.test)):
        # print(i)
        UK = UK_ds.isel(test=i)
        UK_pm25 = UK.pm25.values
        # print(UK.random_sample.values[:2])
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

    modeled, observed = np.ravel(modeled), np.ravel(observed)

    rmse = mean_squared_error(observed, modeled, squared=False)
    print(f"root mean squared error {rmse}")
    mae = mean_absolute_error(observed, modeled)
    print(f"mean absolute error {mean_absolute_error(observed, modeled)}")
    pr = float(pearsonr(observed, modeled)[0])
    print(f"pearson's r {pr}")

    config = str(path.parts[-1])[:-9]
    config = config.replace("Spherical", "")
    config = config.replace("-", " ")
    df = pd.DataFrame(
        {
            "config": [config],
            "frac": str(path.parts[-1])[-5:-3],
            "rmse": [rmse],
            "mae": [mae],
            "pr": [pr],
        }
    )
    return df


# df_list = []
# for path in pathlist:
#     print(int(path.parts[-1])[-5:-3])
# df_list.append(buildsats(path))


df_final = pd.concat([buildsats(path) for path in pathlist]).reset_index()


def plotsns(metric, cmap):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(
        1,
        1,
        1,
    )
    df = df_final[["config", "frac", metric]]
    df = df.pivot("config", "frac", metric)
    sns.heatmap(df, annot=True, ax=ax, cmap=cmap)
    return


plotsns("rmse", cmap="coolwarm")
plotsns("mae", cmap="coolwarm_r")
plotsns("pr", cmap="coolwarm_r")


df = df_final[["config", "frac", "mae"]]
df = df.pivot("config", "frac", "mae")

import plotly.express as px

fig = px.imshow(df)
fig.show()
