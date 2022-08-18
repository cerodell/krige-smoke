Project Overview
-------------------
A comprehensive exam question from **Dr Phil Austin** for **Christopher Rodell**

The project aims to explore kriging and test its varied methods. To explore kriging, we will use air quality observation from locations across North America during a poor air quality event due to high wildfire activity. July 16, 2021, over a thousand active wildfires were burning in western North America. The subsequent smoke from these wildfires produced poor air quality across much of the continent. We will test how well we can interpolate the observed poor air quality to generate a spatial map of ground-level particulate matter 2.5 (PM2.5) concentrations.

Datasets
+++++++++++++



Purple Air
==================================================
- `Purple Air <https://www2.purpleair.com/>`_ is a network of low cost air air quality monitors.
- The air quality metric, Particulate Matter 2.5, will be interpolated. PM 2.5 is tiny particles or droplets in the air that are two and one half microns or less in width.
- Apply Kriging to generate a spatial map of ground-level smoke concentrations across western norther america.
- Data was obtained and reformated using `get-pa-data.py <https://github.com/cerodell/krige-smoke/blob/main/scripts/get-pa-data.py>`_

Government Air Quality Monitors
==================================================
- Government operated air quality stations will be added to the PuprleAir data for the interpolation process
- Data from government operated air quality stations came from multiple sources. More on these data files can be found in the `reformate-gov.py <https://github.com/cerodell/krige-smoke/blob/main/scripts/reformate-gov.py>`_
    - `USA Data <https://aqs.epa.gov/aqsweb/airdata/download_files.html>`_
    - `Alberta Data <https://airdata.alberta.ca/reporting/Download/OneParameter>`_
    - `British Columbia Data <ftp://ftp.env.gov.bc.ca/pub/outgoing/AIR/AnnualSummary/>`_
    - `British Columbia Data <ftp://ftp.env.gov.bc.ca/pub/outgoing/AIR/AnnualSummary/>`_
    - `North West Territories Data <http://aqm.enr.gov.nt.ca/>`_
    - `North West Territories Data <http://aqm.enr.gov.nt.ca/>`_
    - `Manitoba Data <https://web43.gov.mb.ca/EnvistaWeb/Default.ltr.aspx>`_
    - `Ontario Data <http://www.airqualityontario.com/science/data_sets.php>`_
    - `Quebec Data <https://www.environnement.gouv.qc.ca/air/reseau-surveillance/telechargement.asp>`_


Methods
+++++++++++++
   For this exam question we use ordinary kriging (ok), universal kriging with external/specified drift (uk) and regression kriging (rk)

PyKrige
====================================================================================================
- `PyKrige <https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/index.html>`_ is a python library that supports 2D and 3D ordinary, universal and regression kriging.
- PyKrige provides standard variogram models (linear, power, spherical, gaussian, exponential) for the kriging process

.. note::
   The following definitions come from a `geospatial data science course <https://zia207.github.io/geospatial-r-github.io/index.html>`_ created by `Prof. Zia Ahmed <https://www.buffalo.edu/renew/about-us/leadership/zia-ahmed.html>`_ at The State of New York University at Buffalo.

Ordinary kriging (OK)
#########################
Ordinary kriging (OK) is the most widely used kriging method. It is a linear unbiased estimators since error mean is equal to zero. In OK, local mean is filtered from the linear estimator by forcing the kriging weights to sum to 1. The OK is usually preferred to simple kriging because it requires neither knowledge nor stationarity of mean over the entire area

.. math::
   \begin{array}{l}Z_{O K}{ }^{*}(u)=\sum_{\alpha=1}^{n(u)} \lambda_{\alpha}^{O K}(u) Z\left(u_{\alpha}\right) \text { with } \sum_{\alpha=1}^{n(u)} \lambda_{\alpha}^{O K}=1\end{array}


- Where is :math:`\lambda_{\alpha}^{O K}(u)`  weight assigned to the known variables at location  :math:`\left(u_{a}\right)` and :math:`n(u)` is the number of measured values used in estimation of the neighborhood of :math:`u`.


Universal Kriging (UK)
#########################
Universal Kriging (UK) is a variant of the Ordinary Kriging under non-stationary condition where mean differ in a deterministic way in different locations (local trend or drift), while only the variance is constant. This second-order stationarity (“weak stationarity”) is often a pertinent assumption with environmental exposures. In UK, usually first trend is calculated as a function of the coordinates and then the variation in what is left over (the residuals) as a random field is added to trend for making final prediction.

.. math::
   \begin{aligned} Z\left(s_{i}\right) &=m\left(s_{i}\right)+e\left(s_{i}\right) \\ Z(\vec{x}) &=\sum_{k=0}^{K} \beta_{k} f_{k}(\vec{x})+\varepsilon(\vec{x}) \end{aligned}



- Where the :math:`f_{k}` are some global functions of position  :math:`\vec{x}`  and the  :math:`\beta_{k}` are the coefficients.
- The :math:`f` are called base functions.  The  :math:`\varepsilon(\vec{x})`  is the spatially-correlated error, which is modelled as before, with a variogram, but now only considering the residuals, after the global trend is removed.


Regression kriging
#########################
Regression kriging (RK) mathematically equivalent to the universal kriging or kriging with external drift, where auxiliary predictors are used directly to solve the kriging weights. Regression kriging combines a regression model with simple kriging of the regression residuals. The experimental variogram of residuals is first computed and modeled, and then simple kriging (SK) is applied to the residuals to give the spatial prediction of the residuals.

.. math::
   \begin{array}{l}Z_{R K}^{*}(u)=m_{R K}^{*}(u)+\sum_{\alpha=1}^{n(u)} \lambda_{\alpha}^{R K}(u) R\left(u_{\alpha}\right)\end{array}

- Where :math:`m^{*} R K(u \alpha)` is the regression estimate for location :math:`u` and :math:`R(u \alpha)` are the residuals :math:`[R(u \alpha)-m(u \alpha)]` of the observed locations, :math:`n(u)`.


Covariates for UK with external/specified drift.
  #. elevation (take from a `digital elevation model <http://research.jisao.washington.edu/data_sets/elevation/>`_)
  #. aerosol optical depth (derived from the `modis aqua/terra satellites <https://www.nsstc.uah.edu/data/sundar/MODIS_AOD_L3_HRG/>`_)
  #. wind direction (as modeled by `ERA5 <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview>`_)
