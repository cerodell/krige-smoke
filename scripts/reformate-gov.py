# %% [markdown]
# # Air Quality Data
# Used to reformate and combine research grade (government operated) air quality data fro the US and Canada
# Source of data is listed for each location below

# %%
# from unicodedata import name
import context
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path


from context import data_dir, img_dir

## create a list to merge of datasets from all the diff aq datasets
list_ds = []
new_date_range = pd.date_range("2021-07-15T00:00:00", "2021-07-20T00:00:00", freq="H")


# %% [markdown]
# ## USA Data
# - Source: https://aqs.epa.gov/aqsweb/airdata/download_files.html

# %%

try:
    ## open truncated copy of airnow data with dates of interest
    usa_df = pd.read_csv(str(data_dir) + "/obs/usa_trunc.csv")
except:
    ## cant find, create truncated copy of airnow data with dates of interest
    usa_df = pd.read_csv(str(data_dir) + "/obs/hourly_88101_2021.csv")
    usa_df["datetime"] = pd.to_datetime(usa_df["Date GMT"] + "T" + usa_df["Time GMT"])
    usa_df = usa_df.set_index("datetime")
    usa_df = usa_df["2021-07-15T00:00:00":"2021-07-20T00:00:00"]
    usa_df.to_csv(str(data_dir) + "/obs/usa_trunc.csv")

## create a cumston ID for each station to find unique stations
usa_df["IDs"] = usa_df["Latitude"].astype(str) + usa_df["Longitude"].astype(str)
usa_df = usa_df.drop_duplicates(subset=["IDs", "datetime"], keep="first")

## make anew datetime for reindexing aq stations with missing time observations
dfs = dict(tuple(usa_df.groupby("IDs")))

## find unique stations based on created ids
unique_id, count_id = np.unique(usa_df["IDs"], return_counts=True)

## drop out sations that have less than 100 datetime observations..there should be 121 datetime obs
drop_ID = unique_id[count_id > 100]
drop_count = count_id[count_id > 100]

## loop all the stations and make datasets to be merged
for i in range(len(drop_ID)):
    # for i in range(19):
    df = dfs[drop_ID[i]]
    df["datetime"] = pd.to_datetime(df["datetime"])

    ## reindex dataframe for and interpolate missing observation times so all are 121 in length
    df = (
        df.set_index("datetime")
        .reindex(new_date_range)
        .interpolate(method="spline", order=3)
    )

    ## create dataset for each aq station to be merged at the end
    id = f"us_{i}"
    lat = df["Latitude"][0]
    lon = df["Longitude"][0]
    if len(df) > 121:
        print("##############################")
        print(f"US length {len(df)}")
    else:
        pass
    ds = xr.Dataset(
        {
            "PM2.5": (["datetime"], df["Sample Measurement"].values.astype("float32")),
        },
        coords={
            "datetime": df.index.values,
            "id": id,
            "lat": float(lat),
            "lon": float(lon),
        },
    )
    list_ds.append(ds)


# %% [markdown]
# ## AB Data
# - Source: https://airdata.alberta.ca/reporting/Download/OneParameter

# %%
def read_ab(ab_pm25_file):
    ## open ab data files
    ab_df = pd.read_csv(ab_pm25_file, skiprows=5)

    ## make a dictionary for of station aq attributes
    station_dict = {}
    station_dict.update(
        {
            "station_name": [
                s.replace("StationName: ", "") for s in ab_df.iloc[0].values.astype(str)
            ][2:]
        }
    )
    station_dict.update(
        {
            "station_id": [
                s.replace("Station ID: ", "") for s in ab_df.iloc[1].values.astype(str)
            ][2:]
        }
    )
    station_dict.update(
        {
            "station_type": [
                s.replace("Station Type: ", "")
                for s in ab_df.iloc[2].values.astype(str)
            ][2:]
        }
    )
    station_dict.update(
        {
            "station_lat": [
                s.replace("StationLatitude: ", "")
                for s in ab_df.iloc[3].values.astype(str)
            ][2:]
        }
    )
    station_dict.update(
        {
            "station_lon": [
                s.replace("Station Longitude: ", "")
                for s in ab_df.iloc[4].values.astype(str)
            ][2:]
        }
    )
    station_dict.update(
        {
            "station_method": [
                s.replace("Method Name: ", "") for s in ab_df.iloc[6].values.astype(str)
            ][2:]
        }
    )

    ## slice and dice dataframe and make datetime
    ab_df = ab_df.iloc[10:]

    ## make datetime colum and convert from MDT to UTC time
    ab_df["datetime"] = ab_df["Unnamed: 0"].astype("datetime64[ns]") + pd.Timedelta(
        hours=6
    )
    ab_df = ab_df.drop(columns=["Unnamed: 0", "Unnamed: 1"])

    df = ab_df.set_axis(
        station_dict["station_id"] + ["datetime"], axis=1, inplace=False
    )
    df = df.set_index("datetime")
    df = df["2021-07-15T00:00:00":"2021-07-20T00:00:00"]
    df = df.reset_index()

    return df, station_dict


## call on the above function to open and formate ab datafiles
ab_df1, station_dict1 = read_ab(
    str(data_dir) + "/obs/Multiple Stations-One Parameter-2022-06-30 170905.csv"
)
ab_df2, station_dict2 = read_ab(
    str(data_dir) + "/obs/Multiple Stations-One Parameter-2022-06-30 170643.csv"
)

# concatenate dictionary and combine on like keys
ds = [station_dict1, station_dict2]
d = {}
for k in station_dict1.keys():
    d[k] = np.concatenate(list(d[k] for d in ds))

## set up attibutes as arrays to create datasets
ab_ids = d["station_id"]
ab_lats = d["station_lat"]
ab_lons = d["station_lon"]

## combine a dn drop duplicate aq stations
ab_df = ab_df1.merge(ab_df2)
ab_df = ab_df.set_index("datetime")
ab_df = ab_df.loc[:, ~ab_df.columns.duplicated()].copy()

## loop all aq statiosn and create datasets
for column in ab_df:
    ## find the array index of the aq stations based on id
    index = np.where(ab_ids == column)
    try:
        index = index[0][0]
    except:
        index = index[0]
    ## make aq dataset with same formate
    id = f"ab_{index}"
    lat = ab_lats[index]
    lon = ab_lons[index]
    if len(ab_df) > 121:
        print("##############################")
        print(f"AB length {len(ab_df)}")
    else:
        pass
    ds = xr.Dataset(
        {
            "PM2.5": (["datetime"], ab_df[column].values.astype("float32")),
        },
        coords={
            "datetime": ab_df.index.values,
            "id": id,
            "lat": float(lat),
            "lon": float(lon),
        },
    )
    list_ds.append(ds)


# %% [markdown]
# ## BC Data
# - Source: ftp://ftp.env.gov.bc.ca/pub/outgoing/AIR/AnnualSummary/
# %%

try:
    bc_df = pd.read_csv(str(data_dir) + "/obs/bc_aq_trunc.csv")
    bc_df["datetime"] = pd.to_datetime(bc_df["datetime"])
    bc_df = bc_df.set_index("datetime")
except:
    bc_df = pd.read_csv(str(data_dir) + "/obs/bc_aq.csv")
    bc_df["datetime"] = bc_df["DATE_PST"].astype("datetime64[ns]") + pd.Timedelta(
        hours=8
    )
    bc_df = bc_df.set_index("datetime")
    bc_df = bc_df["2021-07-15T00:00:00":"2021-07-20T00:00:00"]
    bc_df.to_csv(str(data_dir) + "/obs/bc_aq_trunc.csv")
    # bc_df = bc_df.drop_duplicates(subset=['EMS_ID'], keep='first')

## make anew datetime for reindexing aq stations with missing time observations
new_date_range = pd.date_range("2021-07-15T00:00:00", "2021-07-20T00:00:00", freq="H")
dfs = dict(tuple(bc_df.groupby("EMS_ID")))

## find unique stations based on created ids
unique_id, count_id = np.unique(bc_df["EMS_ID"], return_counts=True)

## drop out sations that have less than 100 datetime observations..there should be 121 datetime obs
drop_ID = unique_id[count_id == 121]
drop_count = count_id[count_id == 121]

## loop all the stations and make datasets to be merged
for i in range(len(drop_ID)):
    # for i in range(19):
    df = dfs[drop_ID[i]]
    # print(len(df))

    ## reindex dataframe for and interpolate missing observation times so all are 121 in length
    # df = df.reindex(new_date_range).interpolate(method='spline', order=3)

    ## create dataset for each aq station to be merged at the end
    id = f"bc_{i}"
    lat = df["LATITUDE"].iloc[0]
    lon = df["LONGITUDE"].iloc[0]
    if len(df) > 121:
        print("##############################")
        print(f"BC length {len(df)}")
    else:
        pass
    ds = xr.Dataset(
        {
            "PM2.5": (["datetime"], df["RAW_VALUE"].values.astype("float32")),
        },
        coords={
            "datetime": df.index.values,
            "id": id,
            "lat": float(lat),
            "lon": float(lon),
        },
    )
    list_ds.append(ds)


# %% [markdown]
# ## NWT Data
# - Source: http://aqm.enr.gov.nt.ca/
# %%

nwt_df = pd.read_csv(str(data_dir) + "/obs/nwt_aq.csv")
station_name = np.array(list(nwt_df)[2:])
lats = [s for s in nwt_df.iloc[0].values][2:]
lons = [s for s in nwt_df.iloc[1].values][2:]
nwt_df["datetime"] = pd.to_datetime(
    nwt_df["Date"] + "T" + nwt_df["Time"]
) + pd.Timedelta(hours=6)
nwt_df = nwt_df.set_index("datetime")
nwt_df = nwt_df["2021-07-15T00:00:00":"2021-07-20T00:00:00"]
nwt_df = nwt_df.drop(columns=["Date", "Time"])


## loop all the stations and make datasets to be merged
for column in nwt_df:
    ## find the array index of the aq stations based on id
    index = np.where(station_name == column)[0][0]
    ## create dataset for each aq station to be merged at the end
    id = f"nwt_{index}"
    lat = lats[index]
    lon = lons[index]
    if len(nwt_df) > 121:
        print("##############################")
        print(f"NWT length {len(nwt_df)}")
    else:
        pass
    ds = xr.Dataset(
        {
            "PM2.5": (["datetime"], nwt_df[column].values.astype("float32")),
        },
        coords={
            "datetime": nwt_df.index.values,
            "id": id,
            "lat": float(lat),
            "lon": float(lon),
        },
    )
    list_ds.append(ds)


# %% [markdown]
# ## MB Data
# - Source:https://web43.gov.mb.ca/EnvistaWeb/Default.ltr.aspx

# %%
mb_df = pd.read_csv(str(data_dir) + "/obs/mb_aq.csv")
station_name = np.array(list(mb_df)[2:])
lats = [s for s in mb_df.iloc[0].values][2:]
lons = [s for s in mb_df.iloc[1].values][2:]
mb_df["datetime"] = pd.to_datetime(mb_df["Date"] + "T" + mb_df["Time"]) + pd.Timedelta(
    hours=5
)
mb_df = mb_df.set_index("datetime")
mb_df = mb_df["2021-07-15T00:00:00":"2021-07-20T00:00:00"]
mb_df = mb_df.drop(columns=["Date", "Time"])
# mb_df = mb_df.apply(pd.to_numeric).interpolate(method='spline', order=3)

## loop all the stations and make datasets to be merged
for column in mb_df:
    ## find the array index of the aq stations based on id
    index = np.where(station_name == column)[0][0]
    ## create dataset for each aq station to be merged at the end
    id = f"nwt_{index}"
    lat = lats[index]
    lon = lons[index]
    if len(mb_df) > 121:
        print("##############################")
        print(f"MB length {len(mb_df)}")
    else:
        pass
    ds = xr.Dataset(
        {
            "PM2.5": (["datetime"], mb_df[column].values.astype("float32")),
        },
        coords={
            "datetime": mb_df.index.values,
            "id": id,
            "lat": float(lat),
            "lon": float(lon),
        },
    )
    list_ds.append(ds)

# %% [markdown]
# ## OT Data
# - Source: http://www.airqualityontario.com/science/data_sets.php
# %%
id = 0
for i in range(0, 836, 22):
    id += 1
    ot_info = (
        pd.read_csv(str(data_dir) + "/obs/ot_aq.csv", skiprows=i, nrows=11)
        .set_index("From")
        .T
    )
    # station_name = ot_info['Station'].values.astype(str)[0]
    lat = ot_info["Latitude"].values.astype(float)[0]
    lon = ot_info["Longitude"].values.astype(float)[0]
    ot_df = pd.read_csv(str(data_dir) + f"/obs/ot_aq_{id}.csv")
    ot_df["datetime"] = pd.to_datetime(ot_df["datetime"])
    ot_df = ot_df.set_index("datetime")
    # ot_df = pd.read_csv(str(data_dir)+ '/obs/ot_aq.csv', skiprows=12+i, nrows=8, index_col=False)
    # station_id = ot_df['Station ID'].values[0]
    # ot_df = ot_df.drop(columns=['Pollutant', 'Station ID']).T.reset_index()
    # ot_df.columns = ot_df.iloc[0]
    # ot_df = ot_df[1:]

    # keys = [c for c in ot_df if c.startswith('2021-07')]
    # ot_df = ot_df.loc[~ot_df.index.duplicated(keep='first')]
    # ot_df = pd.melt(ot_df, id_vars='Date', value_vars=keys, value_name='pm2.5')
    # ot_df['datetime'] = pd.date_range(f'{ot_df[0].values[0]}T00:00:00',f'{ot_df[0].values[-1]}T023:00:00', freq='H')
    # ot_df = ot_df.set_index('datetime')
    # ot_df = ot_df.drop(columns=['Date', 0])
    # ot_df = ot_df['2021-07-15T00:00:00':'2021-07-20T00:00:00']
    if len(ot_df) > 121:
        print("##############################")
        print(f"MB length {len(ot_df)}")
    else:
        pass
    ds = xr.Dataset(
        {
            "PM2.5": (["datetime"], ot_df["pm2.5"].values.astype("float32")),
        },
        coords={
            "datetime": ot_df.index.values,
            "id": f"ot_{id}",
            "lat": float(lat),
            "lon": float(lon),
        },
    )
    list_ds.append(ds)


# %%

# %% [markdown]
# ## QB Data
# - Source: https://www.environnement.gouv.qc.ca/air/reseau-surveillance/telechargement.asp

with open(str(data_dir) + "/obs/qb_loc.txt") as f:
    qb_locs = f.read().split(",")
qb_locs = [s.replace('"', "") for s in qb_locs]
qb_locs = [s.replace(" ", "") for s in qb_locs]
qb_locs = [s.replace("[", "") for s in qb_locs]
qb_locs = np.array([s.replace("]", "") for s in qb_locs])


qb_fileins = sorted(Path(str(data_dir) + "/obs/").glob(f"PM2,5*"))
i = 0
for filein in qb_fileins:
    i += 1
    qb_df = pd.read_csv(filein, sep=";")
    qb_df["datetime"] = pd.to_datetime(qb_df["Date"]) + pd.Timedelta(hours=4)
    qb_df = qb_df.set_index("datetime")
    qb_df = qb_df["2021-07-15T00:00:00":"2021-07-20T00:00:00"]

    name = qb_df["Nom de la station"][0]
    # print(name)
    index = int(np.where(qb_locs == name.replace(" ", ""))[0])

    ## create dataset for each aq station to be merged at the end
    id = f"us_{i}"
    lat = qb_locs[index + 1]
    lon = qb_locs[index + 2]
    pm25 = list(qb_df["Résultat"].values.astype(str))
    pm25 = [s.replace(",", ".") for s in pm25]
    qb_df["pm25"] = np.array(pm25).astype("float32")
    qb_df = qb_df.reindex(new_date_range).interpolate(method="spline", order=3)
    if len(qb_df) > 121:
        print("##############################")
        print(f"QB length {len(qb_df)}")
    else:
        pass
    ds = xr.Dataset(
        {
            "PM2.5": (["datetime"], qb_df["pm25"].values.astype("float32")),
        },
        coords={
            "datetime": qb_df.index.values,
            "id": id,
            "lat": float(lat),
            "lon": float(lon),
        },
    )

    list_ds.append(ds)

# %% [markdown]
# ## YK Data
# - Source: https://yukon.ca/en/waste-and-recycling/pollution/air-quality-yukon


# %% [markdown]
# ## SK Data
# - Source: http://environment.gov.sk.ca/airqualityindex
# - NO PM2.5 DATA :(

# %%

final_ds = xr.concat(list_ds, dim="id")


def compressor(ds):
    """
    this function compresses datasets
    """
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in ds.data_vars}
    return ds, encoding


ds_concat, encoding = compressor(final_ds)
final_ds.to_netcdf(
    str(data_dir) + f"/gov_aq.nc",
    encoding=encoding,
    mode="w",
)
