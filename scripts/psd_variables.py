import autoroot
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from typing import Callable, Optional
import cmocean
from datetime import datetime
from matplotlib import ticker
import pandas as pd
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)
from dataclasses import dataclass, field
from loguru import logger

from utils.types import generate_wednesdays_in_year
from utils.preprocessing import (
    latlon_deg2m,
    rectilinear_to_regular_grid,
    time_rescale,
    validate_latitude,
    validate_longitude,
    xr_cond_average,
    fuse_base_coords
)
from utils.psd import zonal_lon_psd, space_time_psd

from tqdm.auto import tqdm
from functools import partial
from utils.types import ForecastDataset, DiagnosticDataset, ReAnalysis
from utils.plotting import PlotterContour
from cyclopts import App
import matplotlib.pyplot as plt

from loguru import logger

from distributed import Client
client = Client('scheduler:8786')
logger.log(f"Client: 127.0.0.1:8786 | scheduler:8786")

app = App()


def geoprocess_fn(da, fill_value: int | float | None = None):
    
    # validate coordinates
    da = validate_longitude(validate_latitude(da))
    
    # interpolate to regular grid
    da = rectilinear_to_regular_grid(da, method="linear")
    
    # change lat-lon units (degrees --> Meters)
    da = latlon_deg2m(da)
    
    # change time units (datetime --> days)
    da = time_rescale(da, t0=None, freq_dt=1, freq_unit="D")

    if fill_value is not None:
        da = da.fillna(fill_value)
    
    # sort coordinates
    da = da.sortby("time").sortby("lon").sortby("lat")

    return da

def preprocess_single_leadtime_fn(ds, lead_time: int = 0, idepth: int = 0):
    # select lead time
    try:
        ds = ds.isel(time=lead_time)
    except ValueError:
        pass

    try:
        ds = ds.isel(depth=idepth)
    except ValueError:
        pass
    
    # assign time coord correctly
    ds = ds.assign_coords({"time": np.atleast_1d(ds.time)})
    return ds

def preprocess_all_leadtime_fn(ds, idepth: int = 0):

    # select lead time
    try:
        ds = ds.isel(depth=idepth)
    except ValueError:
        pass
        
    # select the first time step
    t0 =  ds.isel(time=0).time.expand_dims()
    
    # rename time to lead time
    ds = ds.rename({"time": "lead_time"})
    
    # expand dimensions
    ds = ds.expand_dims("time")

    # assign coordinates as time
    ds = ds.assign_coords({"time": np.atleast_1d(t0), "lead_time": np.arange(1,11)})
    return ds

@app.command
def psd_zonal_run_postanalysis(
    model: str = "glo12",
    variable: str = "zos",
    ):

    logger.info(f"Starting Script")
    logger.info(f"Model: {model}")
    logger.info(f"Variable: {variable}")
    dates = list()
    for wednesday in generate_wednesdays_in_year(2024):
        start_datetime = wednesday 
        std_wedn = wednesday.strftime('%Y%m%d')  # Format as YYYYMMDD
        dates.append(std_wedn)
    
    
    # initialize forecast dataset config
    logger.info("Initializing Dataset")
    forecast_config = ForecastDataset() # DiagnosticDataset() # 
    
    demo_model = forecast_config.models[model]
    demo_variable = forecast_config.variables[variable]

    logger.info(f"Loading Paths")
    paths = [str(demo_model.forecast_path(idate)) for idate in dates]
    idepth = 0
    lead_time = 0
    
    fn = partial(preprocess_all_leadtime_fn, idepth=idepth)

    logger.info("Opening Dataset")
    model_results = xr.open_mfdataset(paths, preprocess=fn, combine="by_coords", engine="zarr")


    # select variable
    da = model_results[variable]
    da = demo_variable.correct_real_attrs(da)
    
    # geoprocess variables
    logger.info("Doing Preprocessing")
    fill_value = 0.0
    da = geoprocess_fn(da, fill_value)

    da = da.astype(np.float32)

    logger.info("Running Zonal PSD")
    psd_iso_signal = zonal_lon_psd(da)

    logger.info("Saving Data...")
    psd_iso_signal.name = imodel.name
    path = Path("/home/onyxia/work/OceanBenchFigures/psd/zonal_lon/data")
    path.mkdir(parents=True, exist_ok=True)
    save_name = Path(f"psd_zonallon_global_{ivariable.name}_{imodel.name}_z{idepth}.nc")
    save_name = path.joinpath(save_name)
    logger.debug(f"Saved path: {save_name}")
    psd_iso_signal.to_netcdf(str(save_name))

    logger.success("Finished script!")


@app.command
def psd_spacetime_run_postanalysis(
    model: str = "glo12",
    variable: str = "zos",
    ):

    logger.info(f"Starting Script")
    logger.info(f"Model: {model}")
    logger.info(f"Variable: {variable}")
    dates = list()
    for wednesday in generate_wednesdays_in_year(2024):
        start_datetime = wednesday 
        std_wedn = wednesday.strftime('%Y%m%d')  # Format as YYYYMMDD
        dates.append(std_wedn)
    
    
    # initialize forecast dataset config
    logger.info("Initializing Dataset")
    forecast_config = ForecastDataset() # DiagnosticDataset() # 
    
    demo_model = forecast_config.models[model]
    demo_variable = forecast_config.variables[variable]

    logger.info(f"Loading Paths")
    paths = [str(demo_model.forecast_path(idate)) for idate in dates]
    idepth = 0
    lead_time = 0
    
    fn = partial(preprocess_all_leadtime_fn, idepth=idepth)

    logger.info("Opening Dataset")
    model_results = xr.open_mfdataset(paths, preprocess=fn, combine="by_coords", engine="zarr")


    # select variable
    da = model_results[variable]
    da = demo_variable.correct_real_attrs(da)
    
    # geoprocess variables
    logger.info("Doing Preprocessing")
    fill_value = 0.0
    da = geoprocess_fn(da, fill_value)

    da = da.astype(np.float32)

    logger.info("Running Zonal PSD")
    psd_iso_signal = zonal_lon_psd(da)

    logger.info("Saving Data...")
    psd_iso_signal.name = imodel.name
    path = Path("/home/onyxia/work/OceanBenchFigures/psd/zonal_lon/data")
    path.mkdir(parents=True, exist_ok=True)
    save_name = Path(f"psd_zonallon_global_{ivariable.name}_{imodel.name}_z{idepth}.nc")
    save_name = path.joinpath(save_name)
    logger.debug(f"Saved path: {save_name}")
    psd_iso_signal.to_netcdf(str(save_name))

    logger.success("Finished script!")



if __name__ == "__main__":
    app()
