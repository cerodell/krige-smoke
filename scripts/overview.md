# Background
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
- France/South Africa

#### Why:
- Developed to estimate the most likely distribution of gold based on samples from a few boreholes
- Used when trying to predict the value of a function at a given point. Derived by computing a weighted average of the known values of the function in the neighborhood of the particular point.
- Exploration and Wanderings: Guided by the concept of “not all who wander are lost” (JJ Tolkien)
<hr />

## Methods
Use observational data from a network of low-cost air quality monitors to spatially interpolate PM 2.5 concentrations.




### [Purple Air](https://www2.purpleair.com/)
- PurpleAir is a network of low coast air air quality monitors.
- Use measure Particaul Matter 2.5 data form the PurpleAir network during wildfire smoke events on July 16th 2021.
- Apply Kriging to interpolate ground-level smoke concentrations spatially across western norther america.

### [PyKrige](https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/index.html)
- The code supports 2D and 3D ordinary and universal kriging. Standard variogram models (linear, power, spherical, gaussian, exponential) are built in, but custom variogram models can also be used.
- Test varied methods of kriging
- Try with Basic Methods and expand to Kriging Parameters Tuning, Universal Kriging (ie Regression Kriging)
    - For Universal Kriging convaritate such as surface pressure, 10 meter wind speed and direction and digital elevation models will be tested.
