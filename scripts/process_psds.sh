#!/bin/bash

# Sea Surface Height
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="zos" --model="glonet"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="zos" --model="glo12"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="zos" --model="wenhai"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="zos" --model="xihe"

# Temperature
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="thetao" --model="glonet"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="thetao" --model="glo12"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="thetao" --model="wenhai"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="thetao" --model="xihe"

# Zonal Velocity
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="uo" --model="glonet"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="uo" --model="glo12"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="uo" --model="wenhai"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="uo" --model="xihe"

# Meridional Velocity
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="vo" --model="glonet"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="vo" --model="glo12"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="vo" --model="wenhai"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="vo" --model="xihe"

# Salinity
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="so" --model="glonet"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="so" --model="glo12"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="so" --model="wenhai"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="so" --model="xihe"