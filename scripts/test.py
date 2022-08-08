import context
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import gstools as gs
from context import data_dir
from datetime import datetime

# ids, lat, lon, pm25 = np.loadtxt(os.path.join("..", "data", "pm25_obs.txt")).T

df = pd.read_csv(str(data_dir) + "/obs/gpm25.csv")
lat, lon, pm25 = df["lat"], df["lon"], df["PM2.5"]
###############################################################################
# First we will estimate the variogram of our pm25erature data.
# As the maximal bin distance we choose 8 degrees, which corresponds to a
# chordal length of about 900 km.

bins = gs.standard_bins((lat, lon), max_dist=np.deg2rad(8), latlon=True)
bin_c, vario = gs.vario_estimate((lat, lon), pm25, bin_edges=bins, latlon=True)


model = gs.Spherical(latlon=True, rescale=gs.EARTH_RADIUS)
model.fit_variogram(bin_c, vario, nugget=False)
ax = model.plot("vario_yadrenko", x_max=bin_c[-1])
ax.scatter(bin_c, vario)
ax.set_xlabel("great circle distance / radians")
ax.set_ylabel("semi-variogram")
fig = ax.get_figure()
# fig.savefig(os.path.join("..", "results", "variogram.pdf"), dpi=300)
print(model)

###############################################################################
# As we see, we have a rather large correlation length of ca. 600 km.
#
# Now we want to interpolate the data using Universal and Regression kriging
# in order to compare them.
# We will use a north-south drift by assuming a linear correlation
# of pm25erature with latitude.


def north_south_drift(lat, lon):
    """North south trend depending linearly on latitude."""
    return lat


startTime = datetime.now()
uk = gs.krige.Universal(
    model=model,
    cond_pos=(lat, lon),
    cond_val=pm25,
    drift_functions=north_south_drift,
)
print(f"UK build time {datetime.now() - startTime}")

# fit linear regression model for pm25 depending on latitude
regress = stats.linregress(lat, pm25)
trend = lambda x, y: regress.intercept + regress.slope * x

startTime = datetime.now()
dk = gs.krige.Detrended(
    model=model,
    cond_pos=(lat, lon),
    cond_val=pm25,
    trend=trend,
)
print(f"RK build time {datetime.now() - startTime}")

###############################################################################
# Now we generate the kriging field, by defining a lat-lon grid that covers
# the whole of Germany. The :any:`Krige` class provides the option to only
# krige the mean field, so one can have a glimpse at the estimated drift.

g_lat = np.arange(lat.min(), lat.max() + 2, 1.0)
g_lon = np.arange(lon.min(), lon.max() + 2, 1.0)
startTime = datetime.now()
fld_uk = uk((g_lat, g_lon), mesh_type="structured", return_var=False)
print(f"UK exectue time {datetime.now() - startTime}")

mean = uk((g_lat, g_lon), mesh_type="structured", only_mean=True)

startTime = datetime.now()
fld_dk = dk((g_lat, g_lon), mesh_type="structured", return_var=False)
print(f"RK exectue time {datetime.now() - startTime}")

###############################################################################
# And that's it. Now let's have a look at the generated field and the input
# data along with the estimated mean:

levels = np.linspace(0, 50, 51)
fig, ax = plt.subplots(1, 3, figsize=[10, 5], sharey=True)
sca = ax[0].scatter(lon, lat, c=pm25, vmin=5, vmax=23, cmap="coolwarm")
co1 = ax[1].contourf(g_lon, g_lat, fld_uk, levels, cmap="coolwarm")
co2 = ax[2].contourf(g_lon, g_lat, fld_dk, levels, cmap="coolwarm")

# pdf anti-alias
ax[1].contour(g_lon, g_lat, fld_uk, levels, cmap="coolwarm", zorder=-10)
ax[2].contour(g_lon, g_lat, fld_dk, levels, cmap="coolwarm", zorder=-10)

# [ax[i].plot(border[:, 0], border[:, 1], color="k") for i in range(3)]
# [ax[i].set_xlim([5, 16]) for i in range(3)]
# [ax[i].set_xlabel("Longitude / °") for i in range(3)]
# ax[0].set_ylabel("Latitude / °")

ax[0].set_title("pm25erature observations at 2m\nfrom DWD (2020-06-09 12:00)")
ax[1].set_title("Universal Kriging\nwith North-South drift")
ax[2].set_title("Regression Kriging\nwith North-South trend")

fmt = dict(orientation="horizontal", shrink=0.5, fraction=0.1, pad=0.2)
fig.colorbar(co2, ax=ax, **fmt).set_label("T / °C")
# fig.savefig(os.path.join("..", "results", "kriging.pdf"), dpi=300)

###############################################################################
# To get a better impression of the estimated north-south drift and trend,
# we'll take a look at a cross-section at a longitude of 10 degree:

# fig, ax = plt.subplots()
# label = "latitude-pm25erature scatter"
# reg_trend = trend(g_lat, g_lon)
# ax.scatter(lat, pm25, c="silver", alpha=1.0, edgecolors="none", label=label)
# ax.plot(g_lat, fld_uk[:, 20], c="C0", label="Universal Kriging: pm25erature (10° lon)")
# ax.plot(g_lat, mean[:, 20], "--", c="C0", label="North-South drift: Universal Kriging")
# ax.plot(g_lat, fld_dk[:, 20], c="C1", label="Regression Kriging: pm25erature (10° lon)")
# ax.plot(g_lat, reg_trend, "--", c="C1", label="North-South trend: Regression Kriging")
# ax.set_ylim(7)
# ax.set_xlabel("Latitude / °")
# ax.set_ylabel("T / °C")
# ax.set_title("North-South cross-section")
# ax.legend()
# fig.savefig(os.path.join("..", "results", "trend.pdf"), dpi=300)
