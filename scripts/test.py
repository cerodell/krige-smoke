# import context
# from utils.network import SensorList
# from utils.sensor import Sensor
# from datetime import datetime, timedelta
# import numpy as np
# import pandas as pd
# from pykrige.ok import OrdinaryKriging
# from matplotlib import pyplot as plt
# import gstools as gs


# from pykrige.kriging_tools import write_asc_grid
# import pykrige.kriging_tools as kt
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# from matplotlib.colors import LinearSegmentedColormap
# from matplotlib.patches import Path, PathPatch
# import cartopy.feature as cfeature
# import matplotlib.colors
# from context import data_dir

# p = SensorList()  # Initialized 11,220 sensors!
# # If `sensor_filter` is set to 'column' then we must also provide a value for `column`
# df = p.to_dataframe(sensor_filter='column',
#                     channel='parent',
#                     column='10min_avg')  # See Channel docs for all column options
# print(len(df))


# # wesn = [-137.9,-64.9,24.4,73.3]
# wesn = [-124.8,-117.3,43.6,51.3]

# df = df.loc[(df['lat'] > wesn[2]) & (df['lat'] < wesn[3]) & (df['lon'] > wesn[0]) & (df['lon'] < wesn[1])]
# df = df[df['location_type'] == 'outside']
# df_IDS = pd.DataFrame({'lat': df['lat'].values,'lon': df['lon'].values, 'ids': df.index.values })
# df_IDS.to_csv(str(data_dir)+'/sensorIDs.csv', encoding='utf-8', index=False)


# # np.savetxt(str(data_dir) + '/sensorIDs.txt', ids, fmt='%d')
# # np.savetxt(str(data_dir) + '/sensorLATS.txt', lats, fmt='%d')
# # np.savetxt(str(data_dir) + '/sensorLONS.txt', lons, fmt='%d')

# df = df.loc[(df['10min_avg'] < 100) & (df['10min_avg'] > 0)]
# df = df[df['lat'].notna()]
# df = df[df['lon'].notna()]
# df = df[df['temp_c'].notna()]


# df_filtered = df[df['location_type'] == 'outside']

# lats = df_filtered['lat'].values
# lons = df_filtered['lon'].values
# data = df_filtered['temp_c'].values

# grid_space = 0.1
# grid_lon = np.arange(wesn[0],wesn[1], grid_space)
# grid_lat = np.arange(wesn[2],wesn[3], grid_space)

# OK = OrdinaryKriging(lons, lats, data, variogram_model='gaussian', verbose=True, enable_plotting=False,nlags=20)
# z1, ss1 = OK.execute('grid', grid_lon, grid_lat)

# xintrp, yintrp = np.meshgrid(grid_lon, grid_lat)


# ## bring in state/prov boundaries
# states_provinces = cfeature.NaturalEarthFeature(
#     category="cultural",
#     name="admin_1_states_provinces_lines",
#     scale="50m",
#     facecolor="none",
# )

# # create fig and axes using intended projection
# fig = plt.figure(figsize=(12,10))
# # fig.suptitle(f"Wildfire Conditions", fontsize=22, x=0.48, y = 0.92, weight='bold')
# Cnorm = matplotlib.colors.Normalize(vmin=np.min(z1).astype(int), vmax=np.max(z1).astype(int) + 1)
# # Cnorm = matplotlib.colors.Normalize(vmin=np.min(data).astype(int), vmax=np.max(data).astype(int) + 1)
# # levels = np.arange(np.min(z1).astype(int),np.max(z1).astype(int) + 1, 0.1)
# data_crs = ccrs.PlateCarree()
# ax = fig.add_subplot(1, 1, 1, projection=data_crs)
# CS = ax.contourf(grid_lon, grid_lat, z1, norm=Cnorm, cmap ="jet", levels=levels)
# ax.add_feature(states_provinces, linewidth=0.5, edgecolor="black", zorder=10)
# ax.add_feature(cfeature.BORDERS, zorder=10,  lw = 0.7)
# ax.add_feature(cfeature.COASTLINE, zorder=10, lw = 0.7)
# fig.tight_layout()
# CS = ax.scatter(lons,lats,c = data,norm=Cnorm, cmap ="jet")
# # fig.subplots_adjust(right=0.91)
# # cbar_ax = fig.add_axes([0.92, 0.09, 0.016, 0.38])  # (left, bottom, right, top)
# cbar = fig.colorbar(CS)
# # # draw parallels.
# # parallels = np.arange(21.5,26.0,0.5)
# # m.drawparallels(parallels,labels=[1,0,0,0],fontsize=14, linewidth=0.0) #Draw the latitude labels on the map

# # # draw meridians
# # meridians = np.arange(119.5,122.5,0.5)
# # m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=14, linewidth=0.0)

# # # grid definition for output field
# # gridx = np.arange(0.0, 5.5, 0.1)
# # gridy = np.arange(0.0, 6.5, 0.1)
# # # a GSTools based covariance model
# # cov_model = gs.Gaussian(dim=2, len_scale=4, anis=0.2, angles=-0.5, var=0.5, nugget=0.1)
# # # ordinary kriging with pykrige
# # OK1 = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], cov_model)
# # z1, ss1 = OK1.execute("grid", gridx, gridy)
# # plt.imshow(z1, origin="lower")
# # plt.show()
