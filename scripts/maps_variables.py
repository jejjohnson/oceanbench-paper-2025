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

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)
from dataclasses import dataclass, field
from functools import partial

import matplotlib.pyplot as plt
from cyclopts import App
from loguru import logger
from tqdm.auto import tqdm

from src.plotting import PlotterContour
from src.preprocessing import (fuse_base_coords, latlon_deg2m,
                               rectilinear_to_regular_grid, time_rescale,
                               validate_latitude, validate_longitude,
                               xr_cond_average)
from src.types import (DiagnosticDataset, ForecastDataset, ReAnalysis,
                       generate_wednesdays_in_year)

app = App()


@app.command
def run_map_plotter_forecast(
    variable: str = "zos",
    date: str = "20240103",
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
):
    """
    args:
        variable: str
            input variables, options = ["zos", "uo", "vo", "thetao", "so"]
    """
    logger.info(f"Starting Script...")
    logger.info(f"Calculating Variable: {variable}")

    dates = list()
    for wednesday in generate_wednesdays_in_year(2024):
        start_datetime = wednesday
        std_wedn = wednesday.strftime("%Y%m%d")  # Format as YYYYMMDD
        dates.append(std_wedn)

    len(dates)

    logger.info(f"Initializing GeoProcessing Function...")

    def geoprocess_fn(
        ds, fill_value: int | float | None = None, lead_time: int = 0, idepth: int = 0
    ):

        try:
            ds = ds.isel(depth=idepth, time=lead_time)
        except ValueError:
            ds = ds.isel(time=lead_time)

        # assign time coord correctly
        ds = ds.assign_coords({"time": np.atleast_1d(ds.time)})

        # validate coordinates
        ds = validate_longitude(validate_latitude(ds))

        return ds

    logger.info(f"Loading Datasets...")
    forecast_config = ForecastDataset()  # DiagnosticDataset() #
    variable_config = forecast_config.variables[variable]

    pbar_models = tqdm(forecast_config.models.values(), leave=False)
    idepth = 0
    lead_time = 0
    itime = 0
    fn = partial(geoprocess_fn, idepth=idepth, lead_time=lead_time)

    logger.info(f"Calculating Max Value...")
    if (vmax or vmin) is None:
        with pbar_models:
            max_values = []
            min_values = []
            for imodel in pbar_models:

                pbar_models.set_description(f"Model: {imodel.name}")

                # load paths
                paths = [str(imodel.forecast_path(idate)) for idate in dates]

                # load model results
                model_results = xr.open_mfdataset(
                    paths[:10], preprocess=fn, combine="by_coords", engine="zarr"
                )

                # load the bounds
                max_values.append(
                    model_results.isel(time=itime)[variable].quantile(0.99).values
                )
                min_values.append(
                    model_results.isel(time=itime)[variable].quantile(0.01).values
                )

        # load the bounds
        vmax = np.max(max_values)
        vmin = np.min(min_values)

    logger.info(f"Max Value: {vmax:.2f}")
    logger.info(f"Min Value: {vmin:.2f}")

    pbar_models = tqdm(forecast_config.models.values(), leave=False)

    with pbar_models:
        for imodel in pbar_models:

            pbar_models.set_description(f"Model: {imodel.name}")

            # load paths
            paths = [str(imodel.forecast_path(idate)) for idate in dates]

            # load model results
            model_results = xr.open_mfdataset(
                paths[:10], preprocess=fn, combine="by_coords", engine="zarr"
            )

            model_results = fuse_base_coords(model_results, variable=variable)

            model_results = model_results.sortby("time").sortby("lon").sortby("lat")

            cbar_kwargs = {
                "fraction": 0.02,
                "pad": 0.045,
                "orientation": "vertical",
                # "label": variable_config.label
            }

            try:
                demo_ds = model_results.isel(time=itime, depth=idepth)[variable]
            except:
                demo_ds = model_results.isel(time=itime)[variable]
            demo_ds = demo_ds.sel(lat=slice(-80, 90))
            date = pd.to_datetime(demo_ds.time.values).strftime("%Y-%m-%d")
            ssh_plot = PlotterContour(da=demo_ds, config=variable_config)
            ssh_plot.correct_labels()
            fig, ax = ssh_plot.plot_figure(
                cbar_kwargs=cbar_kwargs, vmin=vmin, vmax=vmax, levels=None
            )
            fig.set(dpi=300)
            path = Path(f"{autoroot.root.joinpath('figures/maps/forecast')}")
            path.mkdir(parents=True, exist_ok=True)
            save_name = Path(
                f"maps_global_{variable}_m{imodel.name}_t{date}_l{itime+1}_z{idepth:.2f}.png"
            )
            save_name = path.joinpath(save_name)
            logger.debug(f"Save Path {imodel.name}: {save_name}")
            fig.savefig(save_name, bbox_inches="tight", transparent=True)
            plt.close()

    logger.info("Plotting the ReAnalysis Data...")
    logger.debug(f"{date} | {type(date)}")
    pd_date = pd.to_datetime(demo_ds.time.values).to_pydatetime()
    logger.debug(f"{pd_date} | {type(pd_date)}")
    model = ReAnalysis()

    depth = 0.49
    data = model.load_data(variable, pd_date, depth)
    data = validate_longitude(validate_latitude(data)).squeeze()[variable]
    data = data.sortby("lon").sortby("lat")
    data = data.sel(lat=slice(-80, 90))
    ssh_plot = PlotterContour(da=data, config=variable_config)
    ssh_plot.correct_labels()
    fig, ax = ssh_plot.plot_figure(
        cbar_kwargs=cbar_kwargs, vmin=vmin, vmax=vmax, levels=None
    )
    fig.set(dpi=300)
    path = Path(f"{autoroot.root.joinpath('figures/maps/forecast')}")
    path.mkdir(parents=True, exist_ok=True)
    save_name = Path(f"maps_global_{variable}_glorys12_t{date}.png")
    save_name = path.joinpath(save_name)
    logger.debug(f"Save Path: {save_name}")
    fig.savefig(path.joinpath(save_name), bbox_inches="tight", transparent=True)
    logger.success(f"Finished Script!")


@app.command
def run_map_plotter_diagnostic(
    variable: str = "u_geo",
    date: str = "20240102",
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
):
    """
    args:
        variable: str
            input variables, options = ["u_geo", "v_geo", "MLD", ]
    """
    logger.info(f"Starting Script...")
    logger.info(f"Calculating Variable: {variable}")

    dates = list()
    for wednesday in generate_wednesdays_in_year(2024):
        start_datetime = wednesday
        std_wedn = wednesday.strftime("%Y%m%d")  # Format as YYYYMMDD
        dates.append(std_wedn)

    len(dates)

    logger.info(f"Initializing GeoProcessing Function...")

    def geoprocess_fn(
        ds, fill_value: int | float | None = None, lead_time: int = 0, idepth: int = 0
    ):

        try:
            ds = ds.isel(depth=idepth, time=lead_time)
        except ValueError:
            ds = ds.isel(time=lead_time)

        # assign time coord correctly
        ds = ds.assign_coords({"time": np.atleast_1d(ds.time)})

        # validate coordinates
        ds = validate_longitude(validate_latitude(ds))

        return ds

    logger.info(f"Loading Datasets...")
    diagnostic_config = DiagnosticDataset()  # DiagnosticDataset() #
    variable_config = diagnostic_config.variables[variable]

    pbar_models = tqdm(diagnostic_config.models.values(), leave=False)
    idepth = 0
    lead_time = 0
    itime = 0
    fn = partial(geoprocess_fn, idepth=idepth, lead_time=lead_time)

    logger.info(f"Calculating Max Value...")
    if (vmax or vmin) is None:
        with pbar_models:
            max_values = []
            min_values = []
            for imodel in pbar_models:

                pbar_models.set_description(f"Model: {imodel.name}")

                # load paths
                paths = [str(imodel.diagnostic_path(idate)) for idate in dates]

                # load model results
                model_results = xr.open_mfdataset(
                    paths[:10], preprocess=fn, combine="by_coords", engine="zarr"
                )

                # load the bounds
                max_values.append(
                    model_results.isel(time=itime)[variable].quantile(0.99).values
                )
                min_values.append(
                    model_results.isel(time=itime)[variable].quantile(0.01).values
                )

        # load the bounds
        vmax = np.max(max_values)
        vmin = np.min(min_values)

    logger.info(f"Max Value: {vmax:.2f}")
    logger.info(f"Min Value: {vmin:.2f}")

    pbar_models = tqdm(forecast_config.models.values(), leave=False)

    with pbar_models:
        for imodel in pbar_models:

            pbar_models.set_description(f"Model: {imodel.name}")

            # load paths
            paths = [str(imodel.diagnostic_path(idate)) for idate in dates]

            # load model results
            model_results = xr.open_mfdataset(
                paths[:10], preprocess=fn, combine="by_coords", engine="zarr"
            )

            model_results = fuse_base_coords(model_results, variable=variable)

            model_results = model_results.sortby("time").sortby("lon").sortby("lat")

            cbar_kwargs = {
                "fraction": 0.02,
                "pad": 0.045,
                "orientation": "vertical",
                # "label": variable_config.label
            }

            try:
                demo_ds = model_results.isel(time=itime, depth=idepth)[variable]
            except:
                demo_ds = model_results.isel(time=itime)[variable]
            demo_ds = demo_ds.sel(lat=slice(-80, 90))
            date = pd.to_datetime(demo_ds.time.values).strftime("%Y-%m-%d")
            ssh_plot = PlotterContour(da=demo_ds, config=variable_config)
            ssh_plot.correct_labels()
            fig, ax = ssh_plot.plot_figure(
                cbar_kwargs=cbar_kwargs, vmin=vmin, vmax=vmax, levels=None
            )
            fig.set(dpi=300)
            path = Path(f"{autoroot.root.joinpath('figures/maps/diagnostic')}")
            path.mkdir(parents=True, exist_ok=True)
            save_name = Path(
                f"maps_global_{variable}_m{imodel.name}_t{date}_l{itime+1}_z{idepth:.2f}.png"
            )
            fig.savefig(path.joinpath(save_name), bbox_inches="tight", transparent=True)
            plt.close()


def load_glorys12_data(
    variable: str,
    date: datetime,
    depth: float,
) -> xr.Dataset:
    """
    Load GLORYS12 data for a specific variable, date, and depth.
    """
    glorys12 = ReAnalysis()
    data = glorys12.load_data(variable, date, depth)
    data = validate_longitude(validate_latitude(data)).squeeze()[variable]
    
    return data

@app.command
def run_map_plotter_forecast_error(
    variable: str = "zos",
    lead_time: int = 0,
    date: str = "20240103",
    vmax: Optional[float] = None,
):
    """
    args:
        variable: str
            input variables, options = ["zos", "uo", "vo", "thetao", "so"]
    """
    logger.info(f"Starting Script...")
    logger.info(f"Calculating Variable: {variable}")
    logger.info(f"Lead Time: {lead_time}")

    dates = list()
    for wednesday in generate_wednesdays_in_year(2024):
        start_datetime = wednesday
        std_wedn = wednesday.strftime("%Y%m%d")  # Format as YYYYMMDD
        dates.append(std_wedn)

    len(dates)

    logger.info(f"Initializing GeoProcessing Function...")

    def geoprocess_fn(
        ds, fill_value: int | float | None = None, lead_time: int = 0, idepth: int = 0
    ):

        try:
            ds = ds.isel(depth=idepth, time=lead_time)
        except ValueError:
            ds = ds.isel(time=lead_time)

        # assign time coord correctly
        ds = ds.assign_coords({"time": np.atleast_1d(ds.time)})

        # validate coordinates
        ds = validate_longitude(validate_latitude(ds))

        return ds

    logger.info(f"Loading Datasets...")
    forecast_config = ForecastDataset()  # DiagnosticDataset() #
    variable_config = forecast_config.variables[variable]
    variable_config.cmap = "Reds"

    pbar_models = tqdm(forecast_config.models.values(), leave=False)
    idepth = 0
    itime = 0
    fn = partial(geoprocess_fn, idepth=idepth, lead_time=lead_time)

    logger.info(f"Calculating Max Value...")
    with pbar_models:
        max_values = []
        base_data = None
        for imodel in pbar_models:

            pbar_models.set_description(f"Model: {imodel.name}")

            # load paths
            path = str(imodel.forecast_path(date))

            compare_model_results = xr.open_mfdataset([path], preprocess=fn, combine="by_coords", engine="zarr").squeeze()[variable]
            
            idate = pd.to_datetime(compare_model_results.time.values).normalize()
            if base_data is None:
                base_data = load_glorys12_data(variable, pd.to_datetime(idate), 0.49)
            
            msg = f"Time mismatch between base data and model results for {imodel.name} on {date}"
            msg += f"\nBase data time: {base_data.time.values}, Model results time: {compare_model_results.time.values}"
            # assert time match to the nearest day
            logger.debug(f"Date: {base_data.time.values} | {compare_model_results.time.values}")
            assert (
                pd.to_datetime(base_data.time.values).normalize()
                ==
                pd.to_datetime(compare_model_results.time.values).normalize()
            ), msg
            # calculate rmse
            diff = (base_data - compare_model_results)
            diff = np.sqrt(diff**2).compute()

            # load the bounds
            max_values.append(np.abs(diff).quantile(0.99).values)

        # load the bounds
        vmax = np.max(max_values)

    logger.info(f"Max Value: {vmax:.2f}")

    pbar_models = tqdm(forecast_config.models.values(), leave=False)

    with pbar_models:
        for imodel in pbar_models:

            pbar_models.set_description(f"Model: {imodel.name}")

            # load paths
            path = str(imodel.forecast_path(date))

            compare_model_results = xr.open_mfdataset([path], preprocess=fn, combine="by_coords", engine="zarr").squeeze()[variable]
            # assert time match to the nearest day
            assert (
                pd.to_datetime(base_data.time.values).normalize()
                ==
                pd.to_datetime(compare_model_results.time.values).normalize()
            ), msg
            # calculate rmse
            diff = (base_data - compare_model_results)
            diff = np.sqrt(diff**2).compute()

            diff = diff.sortby("lon").sortby("lat")

            cbar_kwargs = {
                "fraction": 0.02,
                "pad": 0.045,
                "orientation": "vertical",
                # "label": variable_config.label
            }

            diff = diff.sel(lat=slice(-80, 90))
            
            ssh_plot = PlotterContour(da=diff, config=variable_config)
            ssh_plot.correct_labels()
            fig, ax = ssh_plot.plot_figure(
                cbar_kwargs=cbar_kwargs, vmin=0.0, vmax=vmax, levels=None
            )
            fig.set(dpi=300)
            path = Path(f"{autoroot.root.joinpath('figures/maps/errors/forecast')}")
            path.mkdir(parents=True, exist_ok=True)
            save_name = Path(
                f"maps_global_rmse_{variable}_m{imodel.name}_t{date}_l{lead_time+1}_z{idepth:.2f}.png"
            )
            save_name = path.joinpath(save_name)
            logger.debug(f"Save Path {imodel.name}: {save_name}")
            fig.savefig(save_name, bbox_inches="tight", transparent=True)
            plt.close()

    logger.success(f"Finished Script!")


if __name__ == "__main__":
    app()
