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

# Sea Surface Height
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-diagnostic --variable="u_geo" 
# Temperature
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-diagnostic --variable="v_geo" 
# Zonal Velocity
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-diagnostic --variable="MLD" 
