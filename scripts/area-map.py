from re import S
import context
import numpy as np
import pandas as pd
import xarray as xr

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib as mpl

mpl.rcParams["hatch.linewidth"] = 0.1

import geopandas as gpd
import matplotlib.pyplot as plt
from utils.utils import pixel2poly

from context import data_dir, img_dir


gpm25 = pd.read_csv(str(data_dir) + "/obs/gpm25.csv")
gpm25 = gpd.GeoDataFrame(
    gpm25,
    crs="EPSG:4326",
    geometry=gpd.points_from_xy(gpm25["lon"], gpm25["lat"]),
)

gpm25_poly = gpd.read_file(str(data_dir) + "/obs/outer_bounds")
gpm25_buff = gpd.read_file(str(data_dir) + "/obs/buffer")


gpm25_poly = gpm25_poly.to_crs("EPSG:4326")
gpm25_buff = gpm25_buff.to_crs("EPSG:4326")
gpm25_poly["label"] = ["Study Area"]
gpm25_buff["label"] = ["Verifiaction Area"]

gpm25_verif = gpd.sjoin(gpm25, gpm25_buff, predicate="within")
gpm25_study = gpm25[~gpm25.id.isin(gpm25_verif.id)]


states_provinces = cfeature.NaturalEarthFeature(
    category="cultural",
    name="admin_1_states_provinces_lines",
    scale="50m",
    facecolor="none",
)


legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Test Stations",
        markerfacecolor="tab:green",
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        label="Fixed Stations",
        markerfacecolor="tab:blue",
        markersize=16,
    ),
    Patch(facecolor="lavender", edgecolor="k", label="Verification Area"),
    Patch(facecolor="w", edgecolor="k", hatch="xx", label="Buffer Area"),
]

fig = plt.figure(figsize=(12, 8))
pm25_crs = ccrs.PlateCarree()
ax = fig.add_subplot(111, projection=pm25_crs)
gpm25_poly.plot(
    ax=ax,
    edgecolor="k",
    label=gpm25_poly["label"],
    legend=True,
    facecolor="w",
    hatch="xx",
    zorder=8,
)
gpm25_buff.plot(
    ax=ax, color="lavender", label=gpm25_buff["label"], legend=True, zorder=8
)
ax.scatter(
    gpm25_verif["lon"],
    gpm25_verif["lat"],
    s=40,
    zorder=10,
    color="tab:green",
    edgecolor="k",
    lw=0.4,
)
ax.scatter(
    gpm25_study["lon"],
    gpm25_study["lat"],
    s=40,
    marker="*",
    zorder=10,
    color="tab:blue",
    edgecolor="k",
    lw=0.4,
)
ax.set_xticks(list(np.arange(-130, -80, 10)), crs=ccrs.PlateCarree())
ax.set_yticks(list(np.arange(40, 65, 5)), crs=ccrs.PlateCarree())
ax.tick_params(axis="both", which="major", labelsize=14)
ax.tick_params(axis="both", which="minor", labelsize=14)
ax.add_feature(cfeature.LAND, zorder=1)
ax.add_feature(cfeature.LAKES, zorder=1)
ax.add_feature(cfeature.OCEAN, zorder=1)
ax.add_feature(cfeature.BORDERS, zorder=9)
ax.add_feature(cfeature.COASTLINE, zorder=1)
ax.add_feature(states_provinces, zorder=9)
ax.set_xlabel("Longitude", fontsize=18)
ax.set_ylabel("Latitude", fontsize=18)
# ax.legend(handles=legend_elements, loc='best')

ax.legend(
    handles=legend_elements,
    loc="upper right",
    # bbox_to_anchor=(0.5, 1.08),
    ncol=1,
    fancybox=True,
    shadow=True,
    fontsize=13,
).set_zorder(10)
ax.set_title("Kriging Study Area", fontsize=20, weight="bold")
plt.savefig(str(img_dir) + f"/study-map.png", dpi=300, bbox_inches="tight")
