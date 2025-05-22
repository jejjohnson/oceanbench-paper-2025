from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import autoroot
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

from src.preprocessing import region as region_fn

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)
from dataclasses import dataclass, field
from functools import partial

import matplotlib.pyplot as plt
from cyclopts import App
from distributed import Client
from loguru import logger
from tqdm.auto import tqdm

from src.plotting import PlotterContour
from src.preprocessing import (fuse_base_coords, latlon_deg2m,
                               rectilinear_to_regular_grid, time_rescale,
                               validate_latitude, validate_longitude,
                               xr_cond_average)
from src.psd import isotropic_psd, space_time_psd, zonal_lon_psd
from src.types import (DiagnosticDataset, ForecastDataset, ReAnalysis,
                       generate_wednesdays_in_year)

app = App()


def load_and_preprocess_dataset(
    model: str,
    variable: str,
    region: str,
    year: int = 2024,
    idepth: int = 0,
    fill_value: float = 0.0,
):
    def geoprocess_fn(da, fill_value: int | float | None = None):
        # validate coordinates
        da = validate_longitude(validate_latitude(da))
        da = region_fn(da, region=region)
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
        da = da.astype(np.float32).compute()
        return da

    def preprocess_all_leadtime_fn(ds, idepth: int = 0):
        ds = validate_longitude(validate_latitude(ds))
        ds = region_fn(ds, region=region)
        # select lead time
        try:
            ds = ds.isel(depth=idepth)
        except ValueError:
            pass
        # select the first time step
        t0 = ds.isel(time=0).time.expand_dims()
        # rename time to lead time
        ds = ds.rename({"time": "lead_time"})
        # expand dimensions
        ds = ds.expand_dims("time")
        # assign coordinates as time
        ds = ds.assign_coords(
            {"time": np.atleast_1d(t0), "lead_time": np.arange(1, 11)}
        )
        return ds

    dates = [w.strftime("%Y%m%d") for w in generate_wednesdays_in_year(year)]
    logger.info("Initializing Dataset")
    forecast_config = ForecastDataset()
    demo_model = forecast_config.models[model]
    demo_variable = forecast_config.variables[variable]
    logger.info(f"Loading Paths")
    paths = [str(demo_model.forecast_path(idate)) for idate in dates]
    fn = partial(preprocess_all_leadtime_fn, idepth=idepth)
    logger.info("Opening Dataset")
    model_results = xr.open_mfdataset(
        paths, preprocess=fn, combine="by_coords", engine="zarr"
    )
    da = model_results[variable]
    da = demo_variable.correct_real_attrs(da)
    logger.info("Doing Preprocessing")
    da = geoprocess_fn(da, fill_value)
    return da, idepth


@app.command
def psd_zonal_run_postanalysis(
    model: str = "glo12",
    variable: str = "zos",
    save_path: str | None = None,
    region: str = "gulfstream",
):

    logger.info(f"Starting Script")
    logger.info(f"Model: {model}")
    logger.info(f"Variable: {variable}")
    logger.info(f"Save Path: {save_path}")
    logger.info(f"Region: {region}")

    # Usage in the command:
    da, idepth = load_and_preprocess_dataset(model, variable, region)

    logger.info("Running Zonal PSD")
    psd_iso_signal = zonal_lon_psd(da)

    logger.info("Saving Data...")
    psd_iso_signal.name = model
    if save_path is None:
        path = Path(f"{autoroot.root.joinpath('data/psd/zonallon')}")
    else:
        path = Path(f"{Path(save_path).joinpath('data/psd/zonallon')}")
    path.mkdir(parents=True, exist_ok=True)
    save_name = Path(f"psd_zonallon_{region}_{variable}_{model}_z{idepth}.nc")
    save_name = path.joinpath(save_name)
    logger.debug(f"Saved path: {save_name}")
    psd_iso_signal.to_netcdf(str(save_name))

    logger.success("Finished script!")


@app.command
def psd_zonal_run_plots(
    variable: str = "zos",
    save_path: str | None = None,
    region: str = "gulfstream",
):
    models = ["glorys12", "glo12", "glonet", "wenhai", "xihe"]
    logger.info(f"Starting Script")
    logger.info(f"Variable: {variable}")
    logger.info(f"Save Path: {save_path}")
    logger.info(f"Region: {region}")
    if save_path is None:
        path = Path(f"{autoroot.root.joinpath('data/psd/zonallon')}")
    else:
        path = Path(f"{Path(save_path).joinpath('data/psd/zonallon')}")

    pbar = tqdm(models)

    forecast_config = ForecastDataset()

    idepth = 0
    ilead = 0

    from src.psd import PlotPSDIsotropic

    psd_iso_plot = PlotPSDIsotropic()

    psd_iso_plot.init_fig(figsize=(5, 4))

    with pbar:
        for imodel in pbar:

            # load the data
            save_name = Path(f"psd_zonallon_{region}_{variable}_{imodel}_z{idepth}.nc")
            save_name = path.joinpath(save_name)

            try:

                da = xr.open_dataset(str(save_name), engine="netcdf4")
            except FileNotFoundError:
                logger.error(f"File not found: {save_name}")
                continue

            config = forecast_config.models[imodel]
            da = da[imodel]
            psd_iso_plot.plot_wavelength(
                da.isel(lead_time=ilead),
                freq_scale=1e3,
                units="km",
                label=config.name.upper(),
                color=config.color,
            )

    if save_path is None:
        path = Path(f"{autoroot.root.joinpath('figures/psd/zonallon')}")
    else:
        path = Path(f"{Path(save_path).joinpath('figures/psd/zonallon')}")
    psd_iso_plot.ax.invert_xaxis()
    path.mkdir(parents=True, exist_ok=True)
    save_name = Path(f"psd_{region}_{variable}_z{idepth}_l{ilead}.png")
    save_name = path.joinpath(save_name)
    psd_iso_plot.fig.savefig(save_name, bbox_inches="tight", transparent=True)
    # plt.close()

    logger.success("Finished script!")


@app.command
def psd_spacetime_run_postanalysis(
    model: str = "glo12",
    variable: str = "zos",
    save_path: str | None = None,
    region: str = "gulfstream",
):

    logger.info(f"Starting Script")
    logger.info(f"Model: {model}")
    logger.info(f"Variable: {variable}")
    logger.info(f"Region: {region}")
    logger.info(f"Save Path: {save_path}")

    # Usage in the command:
    da, idepth = load_and_preprocess_dataset(model, variable, region)

    logger.info("Running Zonal PSD")
    psd_st_signal = space_time_psd(da)

    logger.info("Saving Data...")
    psd_st_signal.name = model

    if save_path is None:
        path = Path(f"{autoroot.root.joinpath('data/psd/spacetime')}")
    else:
        path = Path(f"{Path(save_path).joinpath('data/psd/spacetime')}")

    path.mkdir(parents=True, exist_ok=True)
    save_name = Path(f"psd_spacetime_{region}_{variable}_{model}_z{idepth}.nc")
    save_name = path.joinpath(save_name)
    logger.debug(f"Saved path: {save_name}")
    psd_st_signal.to_netcdf(str(save_name))

    logger.success("Finished script!")


@app.command
def psd_spacetime_run_plots(
    variable: str = "zos",
    save_path: str | None = None,
    load_path: str | None = None,
    region: str = "gulfstream",
):
    models = ["glorys12", "glo12", "glonet", "wenhai", "xihe"]
    logger.info(f"Starting Script")
    logger.info(f"Variable: {variable}")
    logger.info(f"Save Path: {save_path}")
    if load_path is None:
        load_path = Path(f"{autoroot.root.joinpath('data/psd/spacetime')}")
    else:
        load_path = Path(f"{Path(save_path).joinpath('data/psd/spacetime')}")

    if save_path is None:
        save_path = Path(f"{autoroot.root.joinpath('figures/psd/spacetime')}")
    else:
        save_path = Path(f"{Path(save_path).joinpath('figures/psd/spacetime')}")
    save_path.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(models)

    forecast_config = ForecastDataset()

    idepth = 0
    ilead = 0

    with pbar:
        for imodel in pbar:

            # load the data
            save_name = Path(f"psd_spacetime_{region}_{variable}_{imodel}_z{idepth}.nc")
            save_name = load_path.joinpath(save_name)

            try:

                da = xr.open_dataset(str(save_name), engine="netcdf4")
            except FileNotFoundError:
                logger.error(f"File not found: {save_name}")
                continue

            from src.psd import PlotPSDSpaceTime

            da = da[imodel]

            psd_st_plot = PlotPSDSpaceTime()
            psd_st_plot.init_fig(figsize=(5, 4))

            psd_st_plot.plot_wavelength(
                da.isel(lead_time=ilead),
                space_scale=1e3,
                space_units="km",
                time_units="days",
                psd_units="SSH",
            )

            file_save_path = save_path.joinpath(
                f"psd_spacetime_{region}_{imodel}_{variable}_z{idepth}_l{ilead}.png"
            )
            psd_st_plot.fig.savefig(
                file_save_path, bbox_inches="tight", transparent=True
            )

    logger.success("Finished script!")


@app.command
def psd_isotropic_run_postanalysis(
    model: str = "glo12",
    variable: str = "zos",
    save_path: str | None = None,
    region: str = "gulfstream",
):

    logger.info(f"Starting Script")
    logger.info(f"Model: {model}")
    logger.info(f"Variable: {variable}")
    logger.info(f"Region: {region}")
    logger.info(f"Save Path: {save_path}")

    # Usage in the command:
    da, idepth = load_and_preprocess_dataset(model, variable, region)

    logger.info("Running Zonal PSD")
    psd_st_signal = isotropic_psd(da)

    logger.info("Saving Data...")
    psd_st_signal.name = model

    if save_path is None:
        path = Path(f"{autoroot.root.joinpath('data/psd/spacetime')}")
    else:
        path = Path(f"{Path(save_path).joinpath('data/psd/spacetime')}")

    path.mkdir(parents=True, exist_ok=True)
    save_name = Path(f"psd_isotropic_{region}_{variable}_{model}_z{idepth}.nc")
    save_name = path.joinpath(save_name)
    logger.debug(f"Saved path: {save_name}")
    psd_st_signal.to_netcdf(str(save_name))

    logger.success("Finished script!")


if __name__ == "__main__":
    app()
