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

from utils.utils import buildsats


pathlistUK = sorted(Path(str(data_dir)).glob(f"UK-*"))
pathlistOK = sorted(Path(str(data_dir)).glob(f"OK-*"))
pathlistRK = sorted(Path(str(data_dir)).glob(f"RK-*"))


pathlist = pathlistOK + pathlistUK + pathlistRK

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
plotsns("mae", cmap="coolwarm")
plotsns("pr", cmap="coolwarm_r")


df = df_final[["config", "frac", "mae"]]
df = df.pivot("config", "frac", "mae")

import plotly.express as px

fig = px.imshow(df)
fig.show()
