# PythonGillAtmosphere
Python implementation of the Gill (1980) atmospheric model, 
with weak temperature gradient extension by Bretherton and Sobel (2003). 

Adapted for Python by Rick Russotto based on Matlab code 
written by Adam Sobel and rewritten by Chris Bretherton.

The analytical model of the atmospheric circulation response to tropical heating 
described by Gill (1980) is a widely used theoretical tool of tropical meteorology.
A Matlab version of the Gill model is available on Github (https://github.com/wy2136/GillModel) 
but no Python version, to our knowledge, has previously been made publicly available online. 
The version posted here also includes the WTG extension described in Bretherton and Sobel (2003). 

# Dependencies/versions

Developed using Python 3.6.6. Imports NumPy version 1.15.0, Matplotlib version 2.2.2,
and several functions from SciPy version 1.1.0.

# How to use

Installation: clone this repository, or just download the "gill.py" file and 
type "import gill" in a Python script or session.

All of the computations and some basic plotting scripts are included in the "Gill.py" module.
To run the model, one must first set up the structure of a heat source (or sink) using the "setupGillM_Gaussian()"
function, which has arguments specifying the grid spacing and dimensions, half-widths, position and scaling of the 
heat source, and whether to subtract the zonal mean from the heat source (required for some applications). 
This function returns a dictionary which is passed to one of two computation functions: 
"GillComputations()" for the classical Gill model, 
or "WTG_Computations()" for the version with the weak temperature gradient approximation applied. 
The computation functions allow the specification of whether or not to run the model in 
an inviscid case (by setting "nodiss" = 1), and return a dict containing the steady state 
divergence, vorticity, zonal and meridional winds, and geopotential height perturbations arising from the 
specified heat source.
Several functions are provided to plot results in a similar format to the Bretherton and Sobel (2003) paper.

Since the model is linear, multiple heat sources can be run individually
and scaled relative to each other via the D0 parameter, and the responses can be 
added to obtain the combined response.

# Test module

Included in the repository is a Jupyter notebook, 
which tests the Gill computation code by reproducing the figures 
in Bretherton and Sobel (2003). This notebook runs the example cases discussed in that paper,
and saves the test figures in a "plots" directory. 
The notebook is also included as an execuatable Python script.
These test cases provide examples of how to use the code and 
provide a way to make sure the code still produces correct results if 
parts of it are changed.

As an example, here is the first figure from the test suite:
![Alt text](plots/BS03_Figure_1.png?raw=true "BS03_Figure_1")

# References

Bretherton, C. S., and A. H. Sobel, 2003: The Gill model and the weak temperature gradient 
approximation. Journal of the Atmospheric Sciences, 60, 451–460, 
doi:10.1175/1520-0469(2003)060<0451:TGMATW>2.0.CO;2.

Gill, A., 1980: Some simple solutions for heat-induced tropical circulation. Quarterly Journal of
the Royal Meteorological Society, 106, 447–462.
