import itertools
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from context import data_dir
import seaborn as sns

sns.set(font_scale=1.4)


def MBE(y_true, y_pred):
    """
    Parameters:
        y_true (array): Array of observed values
        y_pred (array): Array of prediction values

    Returns:
        mbe (float): Biais score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.reshape(len(y_true), 1)
    y_pred = y_pred.reshape(len(y_pred), 1)
    diff = y_true - y_pred
    mbe = diff.mean()
    # print('MBE = ', mbe)
    return mbe


def pixel2poly(x, y, z, resolution):
    """
    x: x coords of cell
    y: y coords of cell
    z: matrix of values for each (x,y)
    resolution: spatial resolution of each cell
    """
    polygons = []
    values = []
    half_res = resolution / 2
    for i, j in itertools.product(range(len(x)), range(len(y))):
        minx, maxx = x[i] - half_res, x[i] + half_res
        miny, maxy = y[j] - half_res, y[j] + half_res
        polygons.append(
            Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])
        )
        if isinstance(z, (int, float)):
            values.append(z)
        else:
            values.append(z[j, i])
    return polygons, values


def plotvariogram(krig):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.plot(krig.lags / 1000, krig.semivariance, "go")
    ax.plot(
        krig.lags / 1000,
        krig.variogram_function(krig.variogram_model_parameters, krig.lags),
        "k-",
    )
    ax.grid(True, linestyle="--", zorder=1, lw=0.5)
    # ax.grid(True)

    try:
        fig_title = f"Coordinates type: {(krig.coordinates_type).title()}" + "\n"
    except:
        fig_title = ""
    if krig.variogram_model == "linear":
        fig_title += "Using '%s' Variogram Model" % "linear" + "\n"
        fig_title += f"Slope: {krig.variogram_model_parameters[0]}" + "\n"
        fig_title += f"Nugget: {krig.variogram_model_parameters[1]}"
    elif krig.variogram_model == "power":
        fig_title += "Using '%s' Variogram Model" % "power" + "\n"
        fig_title += f"Scale:  {krig.variogram_model_parameters[0]}" + "\n"
        fig_title += f"Exponent: + {krig.variogram_model_parameters[1]}" + "\n"
        fig_title += f"Nugget: {krig.variogram_model_parameters[2]}"
    elif krig.variogram_model == "custom":
        print("Using Custom Variogram Model")
    else:
        fig_title += f"Using {(krig.variogram_model).title()} Variogram Model" + "\n"
        fig_title2 = (
            f"Partial Sill: {np.round(krig.variogram_model_parameters[0])}" + "\n"
        )
        fig_title2 += (
            f"Full Sill: {np.round(krig.variogram_model_parameters[0] + krig.variogram_model_parameters[2])}"
            + "\n"
        )
        fig_title2 += (
            f"Range (km): {np.round(krig.variogram_model_parameters[1])/1000}" + "\n"
        )
        fig_title2 += f"Nugget: {np.round(krig.variogram_model_parameters[2],2)}"
    ax.set_title(fig_title, loc="left", fontsize=14)
    # fig_title2 = (
    #     f"Q1 = {np.round(krig.Q1,4)}"
    #     + "\n"
    #     + f"Q2 = {np.round(krig.Q2,4)}"
    #     + "\n"
    #     + f"cR = {np.round(krig.cR,4)}"
    # )
    ax.set_title(fig_title2, loc="right", fontsize=14)

    ax.set_xlabel("Lag (Distance km)", fontsize=12)
    ax.set_ylabel("Semivariance", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)
    return


def buildsats(path):
    # print(str(path))
    gpm25 = pd.read_csv(str(data_dir) + "/obs/gpm25.csv")
    UK_ds = xr.open_dataset(str(path))
    modeled, observed = [], []
    for i in range(len(UK_ds.test)):
        UK = UK_ds.isel(test=i)
        UK_pm25 = UK.pm25.values
        UK_pm25_mean = np.mean(UK_pm25)
        # print(np.unique(np.isnan(UK_pm25)))
        random_sample = gpm25[gpm25.id.isin(UK.random_sample.values)].copy()
        if str(path.parts[-1])[:2] == "RK":
            cordx, cordy = "lon", "lat"
        else:
            cordx, cordy = "Easting", "Northing"

        y = xr.DataArray(
            np.array(random_sample[cordy]),
            dims="ids",
            coords=dict(ids=random_sample.id.values),
        )
        x = xr.DataArray(
            np.array(random_sample[cordx]),
            dims="ids",
            coords=dict(ids=random_sample.id.values),
        )
        pm25_points = UK.pm25.interp(
            x=x, y=y, method="linear", kwargs={"fill_value": UK_pm25_mean}
        )

        random_sample["modeled_PM2.5"] = pm25_points
        modeled.append(pm25_points.values)
        observed.append(random_sample["PM2.5"].values)
        # print(modeled)

    modeled, observed = np.ravel(modeled), np.ravel(observed)

    rmse = mean_squared_error(observed, modeled, squared=False)
    # print(f"root mean squared error {rmse}")
    mae = mean_absolute_error(observed, modeled)
    # print(f"mean absolute error {mean_absolute_error(observed, modeled)}")
    pr = float(pearsonr(observed, modeled)[0])
    # print(f"pearson's r {pr}")
    mbe = MBE(observed, modeled)
    # print(f"mean error (bias) {mbe}")

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
            "mbe": [mbe],
        }
    )
    return df


def buildsatsRK(path):
    # print(str(path))
    gpm25 = pd.read_csv(str(data_dir) + "/obs/gpm25.csv")
    UK_ds = xr.open_dataset(str(path))
    modeled, observed = [], []
    for i in range(len(UK_ds.test)):
        UK = UK_ds.isel(test=i)
        UK_pm25 = UK.pm25.values
        UK_pm25_mean = np.mean(UK_pm25)
        # print(np.unique(np.isnan(UK_pm25)))
        random_sample = gpm25[gpm25.id.isin(UK.random_sample.values)].copy()
        y = xr.DataArray(
            np.array(random_sample["lat"]),
            dims="ids",
            coords=dict(ids=random_sample.id.values),
        )
        x = xr.DataArray(
            np.array(random_sample["lon"]),
            dims="ids",
            coords=dict(ids=random_sample.id.values),
        )
        pm25_points = UK.pm25.interp(
            x=x, y=y, method="linear", kwargs={"fill_value": UK_pm25_mean}
        )

        random_sample["modeled_PM2.5"] = pm25_points
        modeled.append(pm25_points.values)
        observed.append(random_sample["PM2.5"].values)
        # print(modeled)

    modeled, observed = np.ravel(modeled), np.ravel(observed)

    rmse = mean_squared_error(observed, modeled, squared=False)
    # print(f"root mean squared error {rmse}")
    mae = mean_absolute_error(observed, modeled)
    # print(f"mean absolute error {mean_absolute_error(observed, modeled)}")
    pr = float(pearsonr(observed, modeled)[0])
    # print(f"pearson's r {pr}")
    mbe = MBE(observed, modeled)
    # print(f"mean error (bias) {mbe}")

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
            "mbe": [mbe],
        }
    )
    return df


def plotsns(metric, cmap, df_final):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(
        1,
        1,
        1,
    )
    df = df_final[["config", "frac", metric]]
    df = df.pivot("config", "frac", metric)
    if metric == "rmse":
        title = "Root Mean Square Error"
    elif metric == "mae":
        title = "Mean Absolute Error"
    elif metric == "pr":
        title = "Pearson correlation coefficient (r)"
    elif metric == "mbe":
        title = "Mean Error (Bias) "
    else:
        raise ValueError("Not a valid metric option")
    sns.heatmap(df, annot=True, fmt=".4g", ax=ax, cmap=cmap, cbar_kws={"label": title})
    return
