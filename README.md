# Explore Kriging
## Comprehensive exam question
### Dr. Phil Austin
#### Christopher Rodell
<hr />

## Intro Kriging
#### Who:
- French mathematician Georges Matheron (based on the master’s thesis of Danie G. Krige.)

#### What:
- A computationally intensive method of interpolation.
- Commonly referred to as a Gaussian process regression
- A Gaussian process is a probability distribution over possible functions that fit a set of points.

#### When:
- 1960

#### Where:
- France.

#### Why:
- Used when trying to predict the value of a function at a given point. Derived by computing a weighted average of the known values of the function in the neighborhood of the particular point.
- Exploration and Wanderings: Guided by the concept of “not all who wander are lost” (JJ Tolkien)
<hr />

## Methods
### [Purple Air](https://www2.purpleair.com/)
- "Laser pollution sensors by PurpleAir use laser particle counters that provide an accurate and low-cost way to measure smoke, dust, and other particulate air pollution. Laser air quality sensors with both internal storage and data-transmitting abilities."
- Use purple air data during wildfire smoke events.
-

- Apply Kriging to interpolate ground-level smoke concentrations spatially.
### [PyKrige](https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/index.html)
- "The code supports 2D and 3D ordinary and universal kriging. Standard variogram models (linear, power, spherical, gaussian, exponential) are built in, but custom variogram models can also be used."
- Test varied methods of kriging
- Try with Basic Methods and expand to Kriging Parameters Tuning, Regression Kriging,
  - I think these are hyperparameters but need to learn more on this.


### Notes

- external_drift in universal kriging can be the explanatory variables?? something like a dem model, temp or the hms on the same grid etc being predicted by the krig algorithm

- Let’s try nearest neighbour interpolation now using scipy.interpolate.NearestNDInterpolator. As angular coordinates (lat/lon) are not good for measuring distances, I’m going to first convert my data to the linear, meter-based Lambert projection recommend by Statistics Canada and extract the x and y locations as columns in my GeoDataFrame (“Easting” and “Northing” respectively):

- The variogram relates the separating distance between two observation points to a measure of observation similarity at that given distance. Our expectation is that variance is increasing with distance, what can basically be seen in the presented figure.


Variogram models
The last step to describe the spatial pattern in a data set using variograms is to model the empirically observed and calculated experimental variogram with a proper mathematical function. Technically, this setp is straightforward. We need to define a function that takes a distance value and returns a semi-variance value. One big advantage of these models is, that we can assure different things, like positive definitenes. Most models are also monotonically increasing and approach an upper bound. Usually these models need three parameters to fit to the experimental variogram. All three parameters have a meaning and are usefull to learn something about the data. This upper bound a model approaches is called sill. The distance at which 95% of the sill are approached is called the effective range. That means, the range is the distance at which observation values do not become more dissimilar with increasing distance. They are statistically independent. That also means, it doesn’t make any sense to further describe spatial relationships of observations further apart with means of geostatistics. The last parameter is the nugget. It is used to add semi-variance to all values. Graphically that means to move the variogram up on the y-axis. The nugget is the semi-variance modeled on the 0-distance lag. Compared to the sill it is the share of variance that cannot be described spatially.

- So, a good kriging model should result in being 1) Q1 close to zero,
2) Q2 close to one, and 3) cR as small as possible.
