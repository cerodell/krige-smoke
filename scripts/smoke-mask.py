import context
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from context import save_dir

wesn = [-124.8, -117.3, 43.6, 51.3]
# wesn=[-2.0077,2.9159,48.5883,53.4864] # picture frame

# locs={"Paris":[49.2827 -123.1207], "London":[-0.1278,51.5074]}
# nloc=len(locs)

############################################
# TASK 1.1
# Read map
img = mpimg.imread(str(save_dir) + "/2020-09-12T19_00_00Z.png")
# Plot map
plt.figure("channel")
plt.imshow(img, extent=wesn)
plt.xlabel("Longitude [$^\circ$]")
plt.ylabel("Latitude [$^\circ$]")
############################################
# # TASK 1.2 - add labels
# for key in locs.keys():
#     x=float(locs[key][0]); y=float(locs[key][1])
#     plt.scatter(x,y,c='r')
#     plt.annotate(key, xy = (x,y),color='w',xytext = (x,y+0.3),\
#                  bbox = dict(boxstyle="square", fc="black", ec="b",alpha=0.5))

############################################
# TASK 1.3 - smoke mask
# Extract data from individual channels
threshold = 0.2
red = img[:, :, 0]
grn = img[:, :, 1]
blu = img[:, :, 2]
# Copy content over to new array
red1 = red.copy()
grn1 = grn.copy()
blu1 = blu.copy()
# Create smoke mask
cld = np.where((red > threshold) & (grn > threshold) & (blu > threshold))
grn1[cld] = 0
blu1[cld] = 0
msk = np.stack([red1, grn1, blu1], axis=2)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(msk, extent=wesn)  # overlay this image over the original background,
# making it half-transparent
ax.set_title("Threshold value=" + str(threshold))
plt.show()
print("The smoke cover is", round(cld[0].size / red.size * 100.0, 1), "%")


# ############################################
# # TASK 1.4 - histogram
plt.figure("Histogram", figsize=(4, 4))
h, p, s = plt.hist([red, grn, blu], 20, color=["r", "g", "b"])
plt.xlabel("brightness")

############################################
# TASK 1.5 (graduate students) - plot histogram for sub-region.
# Plot map
plt.figure("channel sub-region")
plt.imshow(img, extent=wesn)
plt.xlabel("Longitude [$^\circ$]")
plt.ylabel("Latitude [$^\circ$]")
plt.xlim(0.5, 0.9)
plt.ylim(52.3, 52.7)

# Copy data over for square
red2 = red.copy()
grn2 = grn.copy()
blu2 = blu.copy()

# Square for sub-region (just from mouse-over a certain region)
wesn_sub = [0.651897, 0.716653, 52.4163, 52.4749]
# wesn_sub=[0.651897,0.716653,52.4163,52.4749]

# Assign lat/lon grid to WordVIEW image as done in class
nx = img.shape[1]
ny = img.shape[0]

# Convert lat/lon from wesn_sub into pixel numbers and highlight this region on map
x1 = int((wesn_sub[0] - wesn[0]) / (wesn[1] - wesn[0]) * nx)
x2 = int((wesn_sub[1] - wesn[0]) / (wesn[1] - wesn[0]) * nx)
y1 = ny - int((wesn_sub[3] - wesn[2]) / (wesn[3] - wesn[2]) * ny)
y2 = ny - int((wesn_sub[2] - wesn[2]) / (wesn[3] - wesn[2]) * ny)
grn2[y1:y2, x1:x2] = 1
reg = np.stack([red2, grn2, blu2], axis=2)
# Overlay highlighted region on image
plt.imshow(reg, extent=wesn, alpha=0.5)

#  Make histogram of sub-region
grn2 = grn.copy()
red2 = red2[y1:y2, x1:x2]
grn2 = grn2[y1:y2, x1:x2]
blu2 = blu2[y1:y2, x1:x2]
plt.figure("Histogram sub-region", figsize=(4, 4))
h, p, s = plt.hist([red2, grn2, blu2], 20, color=["r", "g", "b"])
plt.xlabel("brightness")

# Set ranges for scene type as identified from histogram above.
rr = [0.07, 0.17]
gr = [0.15, 0.22]
br = [0.07, 0.17]

# Copy data over for scene type
red3 = red.copy()
grn3 = grn.copy()
blu3 = blu.copy()
frt = np.where(
    (red3 >= rr[0])
    & (red3 <= rr[1])
    & (grn3 >= gr[0])
    & (grn3 <= gr[1])
    & (blu3 >= br[0])
    & (blu3 <= br[1])
)
red3[frt] = 1
grn3[frt] = 0
blu3[frt] = 0
sce = np.stack([red3, grn3, blu3], axis=2)

# Overlay highlighted region on image
plt.figure("Scence type highlighted")
plt.imshow(img, extent=wesn)
plt.imshow(sce, extent=wesn, alpha=1)
plt.show()

############################################
# TASK 2.1 - make scatter plot of blue and green channels vs. the red channels
plt.figure("Scatter")
plt.scatter(red, blu, color="b", s=5)
plt.scatter(red, grn, color="g", s=5, alpha=0.5)
plt.xlabel("Red")
plt.ylabel("Blue and Green")
plt.show()

############################################
# TASK 2.2 - calculate regression coefficient between red and blue brightness
rm = np.mean(red)
bm = np.mean(blu)
rs = np.std(red, ddof=1)
bs = np.std(blu, ddof=1)
# Calculate correlation coefficient explicitly
corx = np.sum((red - rm) * (blu - bm)) / rs / bs / red.size
# Calculate correlation coefficient via built-in function
corb = np.corrcoef(red.flatten(), blu.flatten())[0, 1]
print("Correlation coefficients:", corx, corb)

############################################
# TASK 2.3 - calculate linear fit and determine coefficient of determination
cf, cv = np.polyfit(red.flatten(), blu.flatten(), 1, cov=True)

# from 9/12, slide #6
sst = np.sum((blu - np.mean(blu)) ** 2)
ssr = np.sum((blu - (cf[1] + cf[0] * red)) ** 2)
R2 = 1 - ssr / sst
print("Coefficient of determination:", R2)

xr = np.array([0, 1])
plt.plot(xr, cf[1] + cf[0] * xr, color="b", label="R$^2$=" + str(np.round(R2, 4)))

############################################
# TASK 2.4 - calculate linear fit coefficients from formulae given in lecture
n = red.size
x2 = np.sum(red ** 2)
xx = np.sum(red)
xy = np.sum(red * blu)
yy = np.sum(blu)

D = n * x2 - xx ** 2
A = (x2 * yy - xx * xy) / D
B = (n * xy - xx * yy) / D
sA = bs * np.sqrt(x2 / D)
sB = bs * np.sqrt(n / D)

plt.plot(xr, A + B * xr, ":r", label="from lecture")
plt.legend()

print("Coefficients from polyfit:", cf)
print("Coefficients from lecture:", B, A)
print("Uncertainties            :", sB, sA)

############################################
# TASK 2.5
# First establish color relationship for specific scene type
plt.figure("Color relationship for sub-scene")
plt.scatter(red2, blu2, color="b")
plt.scatter(red2, grn2, color="g")
plt.xlabel("RED")
cb = np.polyfit(red2.flatten(), blu2.flatten(), 1)
cg = np.polyfit(red2.flatten(), grn2.flatten(), 1)
xr = np.array([0.05, 0.25])
plt.plot(xr, cb[1] + cb[0] * xr, color="b")
plt.plot(xr, cg[1] + cg[0] * xr, color="g")

# Now calculate the B and G values that correspond to each R pixel in the domain
B = cb[1] + cb[0] * red
G = cg[1] + cg[0] * red

# Copy data over for scene type
red3 = red.copy()
grn3 = grn.copy()
blu3 = blu.copy()
delta = 0.02
frt = np.where(
    (red3 >= rr[0])
    & (red3 <= rr[1])
    & (np.abs(blu3 - B) < delta)
    & (np.abs(grn3 - G) < delta)
)
red3[frt] = 1
grn3[frt] = 0
blu3[frt] = 0
sce = np.stack([red3, grn3, blu3], axis=2)

# Overlay highlighted region on image
plt.figure("Scence type highlighted NEW method (grad students)")
plt.imshow(img, extent=wesn)
plt.imshow(sce, extent=wesn, alpha=1)
plt.show()
