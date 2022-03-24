import context
from utils.network import SensorList
from utils.sensor import Sensor
from datetime import datetime, timedelta


# p = SensorList()  # Initialized 11,220 sensors!
# # If `sensor_filter` is set to 'column' then we must also provide a value for `column`
# df = p.to_dataframe(sensor_filter='column',
#                     channel='parent',
#                     column='pm_2.5')  # See Channel docs for all column options
# print(len(df))

# nsew = []
dot = datetime(2021, 8, 23)
se = Sensor(2890)
df = se.child.get_historical(weeks_to_get=1, thingspeak_field="primary", start_date=dot)
print(df.head())
