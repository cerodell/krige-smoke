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
- 1960.

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
