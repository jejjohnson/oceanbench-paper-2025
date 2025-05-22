#!/bin/bash
# -----------------------------------------------------------------------------
# Script Name: process_psd_gulf.sh
#
# Description:
#   This script is intended for processing Power Spectral Density (PSD) data
#   specific to the Gulf region as part of the OceanBench 2025 project.
#
# Usage:
#   ./process_psd_gulf.sh
# Args:
#  --save-path=PATH     Path to save the processed data.
#  --region=REGION     Region for processing (default: "gulfstreamsubset").
#   Options: "globe", "zonallon", "gulfstream", "gulfstreamsubset", "northhemisphere"
#
# Notes:
#   - Ensure all required dependencies and environment variables are set before running.
#   - Update the script with specific processing steps as needed.
#
# Author: Juan Emmanuel Johnson
# Date: 23-05-2025
# -----------------------------------------------------------------------------
# Parse keyword arguments: --save-path=... --region=...
for arg in "$@"; do
    case $arg in
        --save-path=*)
            SAVE_PATH="${arg#*=}"
            shift
            ;;
        --region=*)
            REGION="${arg#*=}"
            shift
            ;;
        *)
            ;;
    esac
done

REGION="${REGION:-gulfstreamsubset}"


# Sea Surface Height
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="zos" --model="glonet" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="zos" --model="glo12" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="zos" --model="wenhai" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="zos" --model="xihe" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="zos" --save-path="$SAVE_PATH" --region="$REGION"

PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="zos" --model="glonet" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="zos" --model="glo12" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="zos" --model="wenhai" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="zos" --model="xihe" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-plots --variable="zos" --save-path="$SAVE_PATH" --region="$REGION"

# Temperature
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="thetao" --model="glonet" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="thetao" --model="glo12" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="thetao" --model="wenhai" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="thetao" --model="xihe" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="thetao" --save-path="$SAVE_PATH" --region="$REGION"

PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="thetao" --model="glonet" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="thetao" --model="glo12" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="thetao" --model="wenhai" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="thetao" --model="xihe" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-plots --variable="thetao" --save-path="$SAVE_PATH" --region="$REGION"

# Zonal Velocity
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="uo" --model="glonet" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="uo" --model="glo12" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="uo" --model="wenhai" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="uo" --model="xihe" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="uo" --save-path="$SAVE_PATH" --region="$REGION"

PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="uo" --model="glonet" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="uo" --model="glo12" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="uo" --model="wenhai" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="uo" --model="xihe" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-plots --variable="uo" --save-path="$SAVE_PATH" --region="$REGION"

# Meridional Velocity
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="vo" --model="glonet" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="vo" --model="glo12" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="vo" --model="wenhai" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="vo" --model="xihe" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="vo" --save-path="$SAVE_PATH" --region="$REGION"

PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="vo" --model="glonet" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="vo" --model="glo12" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="vo" --model="wenhai" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="vo" --model="xihe" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-plots --variable="vo" --save-path="$SAVE_PATH" --region="$REGION"

# Salinity
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="so" --model="glonet" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="so" --model="glo12" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="so" --model="wenhai" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-postanalysis --variable="so" --model="xihe" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-zonal-run-plots --variable="so" --save-path="$SAVE_PATH" --region="$REGION"

PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="so" --model="glonet" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="so" --model="glo12" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="so" --model="wenhai" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-postanalysis --variable="so" --model="xihe" --save-path="$SAVE_PATH" --region="$REGION"
PYTHONPATH="." python scripts/psd_variables.py psd-spacetime-run-plots --variable="so" --save-path="$SAVE_PATH" --region="$REGION"

