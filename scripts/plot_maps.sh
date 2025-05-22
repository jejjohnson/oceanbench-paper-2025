#!/bin/bash

# Sea Surface Height
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast --variable="zos" 
# Temperature
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast --variable="thetao" 
# Zonal Velocity
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast --variable="uo" 
# Meridional Velocity
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast --variable="vo" 
# Salinity
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast --variable="so" 
