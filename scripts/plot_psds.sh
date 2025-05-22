#!/bin/bash

# Sea Surface Height
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="zos" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-plots --variable="zos" --save-path="$SAVE_PATH" --region="gulfstreamsubset"

# Temperature
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="thetao" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-plots --variable="thetao" --save-path="$SAVE_PATH" --region="gulfstreamsubset"

# Zonal Velocity
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="uo" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-plots --variable="uo" --save-path="$SAVE_PATH" --region="gulfstreamsubset"

# Meridional Velocity
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="vo" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-plots --variable="vo" --save-path="$SAVE_PATH" --region="gulfstreamsubset"

# Salinity
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="so" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-plots --variable="so" --save-path="$SAVE_PATH" --region="gulfstreamsubset"