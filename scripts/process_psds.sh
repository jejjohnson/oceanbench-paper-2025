#!/bin/bash
SAVE_PATH=${1:-}

# Sea Surface Height
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="zos" --model="glonet" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="zos" --model="glo12" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="zos" --model="wenhai" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="zos" --model="xihe" --save-path="$SAVE_PATH" --region="gulfstreamsubset"

PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="zos" --model="glonet" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="zos" --model="glo12" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="zos" --model="wenhai" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="zos" --model="xihe" --save-path="$SAVE_PATH" --region="gulfstreamsubset"

# Temperature
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="thetao" --model="glonet" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="thetao" --model="glo12" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="thetao" --model="wenhai" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="thetao" --model="xihe" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="thetao" --model="glonet" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="thetao" --model="glo12" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="thetao" --model="wenhai" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="thetao" --model="xihe" --save-path="$SAVE_PATH" --region="gulfstreamsubset"

# Zonal Velocity
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="uo" --model="glonet" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="uo" --model="glo12" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="uo" --model="wenhai" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="uo" --model="xihe" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="uo" --model="glonet" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="uo" --model="glo12" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="uo" --model="wenhai" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="uo" --model="xihe" --save-path="$SAVE_PATH" --region="gulfstreamsubset"

# Meridional Velocity
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="vo" --model="glonet" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="vo" --model="glo12" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="vo" --model="wenhai" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="vo" --model="xihe" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="vo" --model="glonet" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="vo" --model="glo12" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="vo" --model="wenhai" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="vo" --model="xihe" --save-path="$SAVE_PATH" --region="gulfstreamsubset"

# Salinity
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="so" --model="glonet" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="so" --model="glo12" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="so" --model="wenhai" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="so" --model="xihe" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="so" --model="glonet" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="so" --model="glo12" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="so" --model="wenhai" --save-path="$SAVE_PATH" --region="gulfstreamsubset"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="so" --model="xihe" --save-path="$SAVE_PATH" --region="gulfstreamsubset"

