#!/bin/bash

PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast-error --variable="zos" --lead_time=0
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast-error --variable="zos" --lead_time=4
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast-error --variable="zos" --lead_time=9

PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast-error --variable="thetao" --lead_time=0
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast-error --variable="thetao" --lead_time=4
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast-error --variable="thetao" --lead_time=9

PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast-error --variable="uo" --lead_time=0
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast-error --variable="uo" --lead_time=4
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast-error --variable="uo" --lead_time=9

PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast-error --variable="vo" --lead_time=0
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast-error --variable="vo" --lead_time=4
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast-error --variable="vo" --lead_time=9

PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast-error --variable="so" --lead_time=0
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast-error --variable="so" --lead_time=4
PYTHONPATH="." python scripts/maps_variables.py run-map-plotter-forecast-error --variable="so" --lead_time=9