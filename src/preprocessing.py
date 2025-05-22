from typing import Optional, Union, List
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from metpy.calc import lat_lon_grid_deltas
from metpy.units import units
from functools import reduce


def xr_cond_average(
    ds: xr.Dataset, dims: List[str] = None, drop: bool = True
) -> xr.Dataset:
    """Function to conditionally average dimensions

    Args:
        ds (xr.Dataset): the xarray with mixed dimensions
        dims (List[str]): the list of dimensions to average
        drop (bool): option to drop all relevent dimensions

    Returns:
        ds (xr.Dataset): the xr.Dataset with averaged non frequency dimensions
    """

    # get all dims to be conditioned on
    cond_dims = [idim for idim in list(ds.dims) if idim not in dims]

    # create condition
    if len(cond_dims) > 1:
        cond = reduce(lambda x, y: (ds[x] > 0.0) & (ds[y] > 0.0), cond_dims)
    else:
        cond = ds[cond_dims[0]] > 0.0

    # take mean of remaining dims
    return ds.mean(dim=dims).where(cond, drop=drop)


def rectilinear_to_regular_grid(ds, target_grid: xr.Dataset | None = None, **kwargs):

    # Get the bounds
    if not target_grid:
        lat_min, lat_max = ds.lat.min().item(), ds.lat.max().item()
        lon_min, lon_max = ds.lon.min().item(), ds.lon.max().item()
        
        # Approximate resolution (assume uniform spacing)
        dlat = np.abs(ds.lat[1] - ds.lat[0]).mean().item()
        dlon = np.abs(ds.lon[1] - ds.lon[0]).mean().item()
        
        # Create new regular grid
        lat_new = np.arange(lat_min, lat_max + dlat, dlat)
        lon_new = np.arange(lon_min, lon_max + dlon, dlon)
        target_grid = xr.Dataset({'lat': (['lat'], lat_new),
                               'lon': (['lon'], lon_new)})

    # regridder = xe.Regridder(ds, target_grid, method=method)
    # ds_reg = regridder(ds, keep_attrs=True)
    ds_reg = ds.interp_like(target_grid, **kwargs)

    return ds_reg


def latlon_deg2m(ds: xr.Dataset, mean: bool = True) -> xr.Dataset:
    """Converts the lat/lon coordinates from degrees to meters

    Args:
        ds (xr.Dataset): the dataset with the lat/lon variables
        mean (bool): the whether to use the mean dx/dy for each
            lat/lon coordinate (default=True)

    Returns:
        ds (xr.Dataset): the xr.Dataset with the normalized lat/lon coords
    """
    ds = ds.copy()

    lon_attrs = ds["lon"].attrs
    lat_attrs = ds["lat"].attrs

    out = lat_lon_grid_deltas(ds.lon * units.degree, ds.lat * units.degree)

    dx = out[0][:, 0]
    dy = out[1][0, :]

    num_dx = len(dx)
    num_dy = len(dy)

    if mean:
        lat = np.arange(0, num_dx) * np.mean(dx)
        lon = np.arange(0, num_dy) * np.mean(dy)
    else:
        dx0, dy0 = dx[0], dy[0]
        lat = np.cumsum(dx) - dx0
        lon = np.cumsum(dy) - dy0

    lon_attrs.pop("units", None)
    lat_attrs.pop("units", None)

    ds["lon"] = lon
    ds["lat"] = lat
    ds["lon"].attrs = lon_attrs
    ds["lat"].attrs = lat_attrs

    # ds = ds.pint.quantify({"lon": "meter", "lat": "meter"}).pint.dequantify()
    return ds


def time_rescale(
    ds: xr.Dataset,
    freq_dt: int = 1,
    freq_unit: str = "seconds",
    t0: Optional[Union[str, np.datetime64]] = None,
) -> xr.Dataset:
    """Rescales time dimensions of np.datetim64 to an output frequency.

    t' = (t - t_0) / dt

    Args:
        ds (xr.Dataset): the xr.Dataset with a time dimensions
        freq_dt (int): the frequency of the temporal coordinate
        freq_unit (str): the unit for the time frequency parameter
        t0 (datetime64, str): the starting point. Optional. If none, assumes the
            minimum value of the time coordinate

    Returns:
        ds (xr.Dataset): the xr.Dataset with the rescaled time dimensions in the
            freq_unit.
    """

    ds = ds.copy()

    if t0 is None:
        t0 = ds["time"].min()

    if isinstance(t0, str):
        t0 = np.datetime64(t0)

    ds["time"] = ((ds["time"] - t0) / pd.to_timedelta(freq_dt, unit=freq_unit)).astype(
        np.float32
    )

    # ds = ds.pint.quantify({"time": freq_unit}).pint.dequantify()

    return ds


def validate_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Format lat and lon variables

    Set units, ranges and names

    Args:
        ds: input data

    Returns:
        formatted data
    """
    new_ds = ds.copy()

    new_ds = _rename_longitude(new_ds)

    ds_attrs = new_ds.lon.attrs

    new_ds["lon"] = transform_360_to_180(new_ds.lon)
    new_ds["lon"] = new_ds.lon.assign_attrs(
        **{
            **ds_attrs,
            **dict(
                units="degrees_east",
                standard_name="longitude",
                long_name="Longitude",
            ),
            
        }
    )

    return new_ds


def validate_latitude(ds: xr.Dataset) -> xr.Dataset:

    new_ds = ds.copy()

    new_ds = _rename_latitude(new_ds)

    ds_attrs = new_ds.lat.attrs

    new_ds["lat"] = transform_180_to_90(new_ds.lat)
    new_ds["lat"] = new_ds.lat.assign_attrs(
        **{
            **ds_attrs,
            **dict(
                units="degrees_north",
                standard_name="latitude",
                long_name="Latitude",
            ),
        }
    )

    return new_ds

def fuse_base_coords(ds, variable: str,):
    # fix the MLD coordinates...
    try:
        ds[variable] = ds.drop_vars(["latitude", "longitude"])[variable].rename({"latitude": "lat", "longitude": "lon"})
        ds = ds.drop_vars(["latitude", "longitude"])
    except ValueError:
        pass
    return ds

def _rename_longitude(ds):
    try:
        ds = ds.rename({"longitude": "lon"})
    except:
        pass
    return ds

def _rename_latitude(ds):
    try:
        ds = ds.rename({"latitude": "lat"})
    except:
        pass
    return ds


def transform_360_to_180(coord: np.ndarray) -> np.ndarray:
    """
    This function converts the coordinates that are bounded from [-180, 180]
    to coordinates bounded by [0, 360].

    Args:
        coord (np.ndarray): The input array of coordinates.

    Returns:
        np.ndarray: The output array of coordinates.
    """
    return (coord + 180) % 360 - 180


def transform_180_to_90(coord: np.ndarray) -> np.ndarray:
    """
    This function converts the coordinates that are bounded from [-180, 180]
    to coordinates bounded by [0, 360].

    Args:
        coord (np.ndarray): The input array of coordinates.

    Returns:
        np.ndarray: The output array of coordinates.
    """
    return (coord + 90) % 180 - 90