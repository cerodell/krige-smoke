import context
import numpy as np
import pandas as pd
from utils.network import SensorList
from utils.sensor import Sensor
from datetime import datetime, timedelta
from context import data_dir
import xarray as xr


# %% [markdown]
# # Inputs
# Select data range of interest and open csv flies for location of interest
# %%
start = datetime(2021, 7, 15)
stop = datetime(2021, 7, 20)

df_IDS = pd.read_csv((str(data_dir) + "/sensorIDs_north_america.csv"))
# df_IDS = df_IDS[0:2]

# %% [markdown]
# # Get Data
# Get data for location and time of interest
# %%

start_stop = pd.date_range(
    start.strftime("%Y-%m-%dT%H:%M:%S"), stop.strftime("%Y-%m-%dT%H:%M:%S"), freq="10T"
)

validID, validDF, validLAT, validLON, flagIDS = [], [], [], [], []
for i in range(len(df_IDS)):
    id = df_IDS["ids"][i]
    lat = df_IDS["lat"][i]
    lon = df_IDS["lon"][i]
    try:
        se = Sensor(int(id))
        df = se.child.get_historical(
            weeks_to_get=1,
            thingspeak_field="primary",
            start_date=start + timedelta(days=14),
        )
        if len(df) < 1:
            pass
        else:
            # print('Found DF')
            df = df.resample("10Min", on="created_at").mean()
            df = df[
                (
                    df.index
                    > (start - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S")
                )
                & (df.index <= stop.strftime("%Y-%m-%dT%H:%M:%S"))
            ]
            x, x_counts = np.unique(
                np.isnan(df["PM2.5 (CF=1) ug/m3"].values), return_counts=True
            )
            print(x)
            print(x_counts)
            if len(df) != len(start_stop):
                pass
            else:
                try:
                    if (x_counts[1] > 100) and (x[1] == True):
                        pass
                    else:
                        validID.append(id)
                        print(
                            f"ID {id} valid with df length {len(df)} with {x_counts[1]} NaNs"
                        )
                        validDF.append(df)
                        validLAT.append(lat)
                        validLON.append(lon)
                        flagIDS.append(id)
                except:
                    validID.append(id)
                    print(f"ID {id} valid with df length {len(df)}")
                    validDF.append(df)
                    validLAT.append(lat)
                    validLON.append(lon)
    except:
        pass


# blah = np.unique(np.isnan(validDF[0]['PM2.5 (CF=1) ug/m3'].values), return_counts=True)
# for i in range(len(validDF2)):
#   print(i)
#   if len(np.unique(np.isnan(validDF2[i]['PM2.5 (CF=1) ug/m3'].values))) == 1:
#     print('Pass')
#     print(np.unique(np.isnan(validDF2[i]['PM2.5 (CF=1) ug/m3'].values))[0]==False)
#   else:
#     print('Fail')


list_ds = []
for i in range(len(validDF)):
    df = validDF[i]
    id = validID[i]
    lat = validLAT[i]
    lon = validLON[i]
    ds = xr.Dataset(
        {
            "PM1.0": (["datetime"], df["PM1.0 (CF=1) ug/m3"].values.astype("float32")),
            "PM2.5": (["datetime"], df["PM2.5 (CF=1) ug/m3"].values.astype("float32")),
            "PM10.0": (
                ["datetime"],
                df["PM10.0 (CF=1) ug/m3"].values.astype("float32"),
            ),
            "pressure": (
                ["datetime"],
                df["Atmospheric Pressure"].values.astype("float32"),
            ),
            "PM2.5_ATM": (
                ["datetime"],
                df["PM2.5 (CF=ATM) ug/m3"].values.astype("float32"),
            ),
        },
        coords={
            "datetime": df.index.values,
            "id": id,
            "lat": lat,
            "lon": lon,
        },
    )
    list_ds.append(ds)


ds_concat = xr.concat(list_ds, dim="id")


def compressor(ds):
    """
    this function compresses datasets
    """
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    return ds, encoding


ds_concat, encoding = compressor(ds_concat)
ds_concat.to_netcdf(
    str(data_dir) + f"/purpleair_north_america.nc",
    encoding=encoding,
    mode="w",
)
