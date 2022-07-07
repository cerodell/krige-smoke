import context

import numpy as np
import pandas as pd
import plotly.express as px

from utils.sensor import Sensor
from utils.network import SensorList

from context import data_dir


p = SensorList()  # Initialized sensors!

# If `sensor_filter` is set to 'column' then we must also provide a value for `column`
df = p.to_dataframe(
    sensor_filter="column", channel="parent", column="10min_avg"
)  # See Channel docs for all column options
print(len(df))

wesn = [-160.0, -52.0, 32.0, 70.0]
# var forecastBounds = [[32.0, -160.0], [70.0, -52.0]];


df = df.loc[
    (df["lat"] > wesn[2])
    & (df["lat"] < wesn[3])
    & (df["lon"] > wesn[0])
    & (df["lon"] < wesn[1])
]
df = df[df["location_type"] == "outside"]
df_IDS = pd.DataFrame(
    {"lat": df["lat"].values, "lon": df["lon"].values, "ids": df.index.values}
)
df_IDS.to_csv(
    str(data_dir) + "/sensorIDs_north_america.csv", encoding="utf-8", index=False
)
# %%
# px.set_mapbox_access_token(open(".mapbox_token").read())
fig = px.scatter_geo(
    df_IDS, lat=df_IDS["lat"].values, lon=df_IDS["lon"].values, hover_name="ids"
)
fig.update_layout(
    geo=dict(
        scope="north america",
        fitbounds="locations",
        showland=True,
        landcolor="rgb(212, 212, 212)",
        subunitcolor="rgb(255, 255, 255)",
        countrycolor="rgb(255, 255, 255)",
        showlakes=True,
        lakecolor="rgb(255, 255, 255)",
        showsubunits=True,
        showcountries=True,
        resolution=50,
        projection=dict(type="conic conformal", rotation_lon=-100),
        lonaxis=dict(showgrid=True, gridwidth=0.5, range=[-140.0, -55.0], dtick=5),
        lataxis=dict(showgrid=True, gridwidth=0.5, range=[20.0, 60.0], dtick=5),
    ),
    title='Air Quality Monitors <br>Source: <a href="https://www2.purpleair.com/">PurpleAir</a>',
)
fig.show()
# %%
