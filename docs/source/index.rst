.. Explore Kriging documentation master file, created by
   sphinx-quickstart on Wed Jul 13 11:18:24 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


..  Explore Kriging
.. ====================

Explore Kriging
******************************************************


Overview
-------------------
A comprehensive exam question from **Dr Phil Austin** for **Christopher Rodell**

Goal of project is to explore kriging and test its varied methods.

Kriging Background
++++++++++++++++++++



   "`Kriging <https://zia207.github.io/geospatial-r-github.io/kriging.html>`_ is a group of geostatistical techniques to interpolate the value of a random field (e.g., the soil variables as a function of the geographic location) at an un-sampled location from known observations of its value at nearby locations. The main statistical assumption behind kriging is one of stationarity which means that statistical properties (such as mean and variance) do not depend on the exact spatial locations, so the mean and variance of a variable at one location is equal to the mean and variance at another location. The basic idea of kriging is to predict the value at a given point by computing a weighted average of the known values of the function in the neighborhood of the point. Unlike other deterministic interpolation methods such as inverse distance weighted (IDW) and Spline, kriging is based on auto-correlation-that is, the statistical relationships among the measured points to interpolate the values in the spatial field. Kriging is capable to produce prediction surface with uncertainty. Although stationarity (constant mean and variance) and isotropy (uniformity in all directions) are the two main assumptions for kriging to provide best linear unbiased prediction, however, there is flexibility of these assumptions for various forms and methods of kriging." - `Prof. Zia Ahmed <https://www.buffalo.edu/renew/about-us/leadership/zia-ahmed.html>`_



Who
=====
- Developed by french mathematician Georges Matheron (based on the master’s thesis of Danie G. Krige.)

What
=====
- A computationally intensive method of interpolation.
- Commonly referred to as a Gaussian process regression
- A Gaussian process is a probability distribution over possible functions that fit a set of points.


When
==========
- 1960

Where
=======
- France/South Africa

Why
=====
- Developed to estimate the most likely distribution of gold based on samples from a few boreholes
- Used when trying to predict the value of a function at a given point. Derived by computing a weighted average of the known values of the function in the neighborhood of the particular point.
- Exploration and Wanderings: Guided by the concept of “not all who wander are lost” (JJ Tolkien)



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   comps-proj
   comps-data
   comps-ok
   comps-uk-bsp
   comps-rk-dem
   comps-ver




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
