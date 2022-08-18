import context
import numpy as np
import pandas as pd
import xarray as xr
import salem

import gstools as gs
from scipy import stats
from datetime import datetime


import geopandas as gpd
import plotly.graph_objects as go


from context import data_dir, img_dir
import time

start_time = time.time()

# %%

frac = 0.10


wesn = [-129.0, -90.0, 40.0, 60.0]  ## Big Test Domain
resolution = 0.25  # grid cell size in degress

dem_ds = salem.open_xr_dataset(str(data_dir) + f"/elev.americas.5-min.nc")
dem_ds["lon"] = dem_ds["lon"] - 360

gov_ds = xr.open_dataset(str(data_dir) + f"/gov_aq.nc")
gov_ds = gov_ds.sel(datetime="2021-07-16T22:00:00")

pa_ds = xr.open_dataset(str(data_dir) + f"/purpleair_north_america.nc")
pa_ds = pa_ds.sel(datetime="2021-07-16T22:00:00")
pa_ds = pa_ds.drop(["PM1.0", "PM10.0", "pressure", "PM2.5_ATM"])

ds = xr.concat([pa_ds, gov_ds], dim="id")


df = pd.read_csv(str(data_dir) + "/obs/gpm25.csv")
gpm25 = gpd.GeoDataFrame(
    df,
    crs="EPSG:4326",
    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
).to_crs("EPSG:3347")
gpm25["Easting"], gpm25["Northing"] = gpm25.geometry.x, gpm25.geometry.y
gpm25.head()
# gpm25.to_csv(str(data_dir) + "/obs/gpm25.csv")

# %%

## make grid based on dataset bounds and resolution
g_lon = np.arange(
    df["lon"].min() - resolution,
    df["lon"].max() + resolution,
    resolution,
)
g_lat = np.arange(
    df["lat"].min() - resolution,
    df["lat"].max() + resolution,
    resolution,
)


## use salem to create a dataset with the grid.
krig_ds = salem.Grid(
    nxny=(len(g_lon), len(g_lat)),
    dxdy=(resolution, resolution),
    x0y0=(df["lon"].min(), df["lat"].min()),
    proj="EPSG:4326",
    pixel_ref="corner",
).to_dataset()
## print dataset
krig_ds


def dem_points(df):
    y = xr.DataArray(
        np.array(df["lat"]),
        dims="ids",
        coords=dict(ids=df.id.values),
    )
    x = xr.DataArray(
        np.array(df["lon"]),
        dims="ids",
        coords=dict(ids=df.id.values),
    )
    var_points = dem_ds.data.interp(lon=x, lat=y, method="linear").values[0, :]
    if len(df.index) == len(var_points):
        pass
    else:
        raise ValueError("Lenghts dont match")
    return var_points
    # df["dem"] = dem_points.data[0, :]
    # return df["dem"].values


def compressor(ds):
    """
    this function compresses datasets
    """
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    return ds, encoding


for frac in [0.1, 0.3, 0.5]:
    print(f"looping {int(frac*100)}")
    random_sample_df = pd.read_csv(
        str(data_dir) + f"/random-samples-{int(frac*100)}.csv", index_col=0
    )
    list_ds, random_ids_list = [], []
    for i in range(0, 10):
        loopTime = datetime.now()

        ds = krig_ds
        random_ids = random_sample_df[str(i)].values
        # print(random_ids)
        gpm25_krig = gpm25.loc[~gpm25.id.isin(random_ids)]
        lat, lon, pm25 = gpm25_krig["lat"], gpm25_krig["lon"], gpm25_krig["PM2.5"]
        bins = gs.standard_bins((lat, lon), max_dist=np.deg2rad(8), latlon=True)
        bin_c, vario = gs.vario_estimate((lat, lon), pm25, bin_edges=bins, latlon=True)
        model = gs.Spherical(latlon=True, rescale=gs.EARTH_RADIUS)
        para, pcov, r2 = model.fit_variogram(bin_c, vario, nugget=False, return_r2=True)
        print(r2)

        print(f"Random Sample index 0 {random_ids[0]}")
        dem = dem_points(gpm25_krig)
        startTime = datetime.now()
        # fit linear regression model for pm25 depending on aod
        regress = stats.linregress(dem, pm25)
        trend = lambda x, y: regress.intercept + regress.slope * x

        startTime = datetime.now()
        dk = gs.krige.Detrended(
            model=model,
            cond_pos=(lat, lon),
            cond_val=pm25.values,
            trend=trend,
        )
        print(f"RK build time {datetime.now() - startTime}")

        startTime = datetime.now()
        fld_dk = dk((g_lat, g_lon), mesh_type="structured", return_var=False)
        print(f"RK exectue time {datetime.now() - startTime}")
        RK_pm25 = np.where(fld_dk < 0, 0, fld_dk)

        ds.assign_coords({"test": i})
        ds.assign_coords({"ids": np.arange(len(random_ids))})
        ds["pm25"] = (("y", "x"), RK_pm25)
        random_ids_list.append(random_ids.astype(str))

        list_ds.append(ds)
        print(f"Loop {i} time {datetime.now() - loopTime}")

    final_ds = xr.concat(list_ds, dim="test")
    final_ds["random_sample"] = (("test", "ids"), np.stack(random_ids_list))
    final_ds["random_sample"] = final_ds["random_sample"].astype(str)

    ds_concat, encoding = compressor(final_ds)
    final_ds.to_netcdf(
        str(data_dir) + f"/RK-dem-{int(frac*100)}.nc",
        encoding=encoding,
        mode="w",
    )
