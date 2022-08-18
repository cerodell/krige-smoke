import context
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import gstools as gs
from context import data_dir

###############################################################################
# Generate a synthetic field with an exponential model.

# x = np.random.RandomState(19970221).rand(1000) * 100.0
# y = np.random.RandomState(20011012).rand(1000) * 100.0
# model = gs.Exponential(dim=2, var=2, len_scale=8)
# srf = gs.SRF(model, mean=0, seed=19970221)
# field = srf((x, y))

df = pd.read_csv(str(data_dir) + "/obs/gpm25.csv")
y, x, field = df["lon"], df["lat"], df["PM2.5"]
y, x, field = df["Easting"], df["Northing"], df["PM2.5"]

###############################################################################
# Estimate the variogram of the field with 40 bins and plot the result.

# bins = np.arange(10)
# bin_center, gamma = gs.vario_estimate((x, y), field, bins, latlon=False)

# bin_center, gamma = gs.vario_estimate((x, y), field)
# print("estimated bin number:", len(bin_center))
# print("maximal bin distance:", max(bin_center))


# bins = gs.standard_bins((x, y), max_dist=np.deg2rad(8), latlon=True)
# bin_center, gamma = gs.vario_estimate((x, y), field, bin_edges=bins, latlon=True)

bins = gs.standard_bins((x, y), latlon=False)
bin_center, gamma = gs.vario_estimate((x, y), field, bin_edges=bins, latlon=False)
###############################################################################
# Define a set of models to test.

models = {
    "Gaussian": gs.Gaussian,
    "Exponential": gs.Exponential,
    "Matern": gs.Matern,
    "Stable": gs.Stable,
    "Rational": gs.Rational,
    "Circular": gs.Circular,
    "Spherical": gs.Spherical,
    "SuperSpherical": gs.SuperSpherical,
    "JBessel": gs.JBessel,
}
scores = {}

###############################################################################
# Iterate over all models, fit their variogram and calculate the r2 score.

# plot the estimated variogram
plt.scatter(bin_center, gamma, color="k", label="data")
ax = plt.gca()

# fit all models to the estimated variogram
# for model in models:
#     fit_model = models[model](latlon=True, rescale=gs.EARTH_RADIUS)
#     para, pcov, r2 = fit_model.fit_variogram(bin_center, gamma, return_r2=True, nugget=False)
#     fit_model.plot(x_max=bin_center[-1], ax=ax)
#     scores[model] = r2


for model in models:
    fit_model = models[model](latlon=False)
    para, pcov, r2 = fit_model.fit_variogram(
        bin_center, gamma, return_r2=True, nugget=False
    )
    fit_model.plot(x_max=bin_center[-1], ax=ax)
    scores[model] = r2

###############################################################################
# Create a ranking based on the score and determine the best models

ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
print("RANKING by Pseudo-r2 score")
for i, (model, score) in enumerate(ranking, 1):
    print(f"{i:>6}. {model:>15}: {score:.5}")

plt.show()
