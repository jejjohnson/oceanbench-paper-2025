# SEA SURFACE HEIGHTR
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="zos" --save-path="$SAVE_PATH" --region="gulfstreamsubset" --lead_time=0
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="zos" --save-path="$SAVE_PATH" --region="gulfstreamsubset" --lead_time=4
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="zos" --save-path="$SAVE_PATH" --region="gulfstreamsubset" --lead_time=9

# TEMPERATURE
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="thetao" --save-path="$SAVE_PATH" --region="gulfstreamsubset" --lead_time=0
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="thetao" --save-path="$SAVE_PATH" --region="gulfstreamsubset" --lead_time=4
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="thetao" --save-path="$SAVE_PATH" --region="gulfstreamsubset" --lead_time=9

# ZONAL VELOCITY
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="uo" --save-path="$SAVE_PATH" --region="gulfstreamsubset" --lead_time=0
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="uo" --save-path="$SAVE_PATH" --region="gulfstreamsubset" --lead_time=4
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="uo" --save-path="$SAVE_PATH" --region="gulfstreamsubset" --lead_time=9

# MERIDIONAL VELOCITY
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="vo" --save-path="$SAVE_PATH" --region="gulfstreamsubset" --lead_time=0
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="vo" --save-path="$SAVE_PATH" --region="gulfstreamsubset" --lead_time=4
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="vo" --save-path="$SAVE_PATH" --region="gulfstreamsubset" --lead_time=9

# SALINITY
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="so" --save-path="$SAVE_PATH" --region="gulfstreamsubset" --lead_time=0
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="so" --save-path="$SAVE_PATH" --region="gulfstreamsubset" --lead_time=4
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="so" --save-path="$SAVE_PATH" --region="gulfstreamsubset" --lead_time=9