from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
import warnings

from typing import List
import xrft
import xarray as xr
from functools import reduce
import seaborn as sns
from dask.diagnostics import ProgressBar
from .preprocessing import xr_cond_average

def zonal_lon_psd(da: xr.DataArray) -> xr.DataArray:
    with ProgressBar():
    
        # frequency dims
        psd_iso_signal = xrft.power_spectrum(
            da.chunk({
                "time": 1,
                "lon": da.lon.shape[0], 
                "lat": 1,}),
            dim=["lon", ],
            detrend="linear",
            window="tukey",
            nfactor=2,
            window_correction=True,
            true_amplitude=True,
            truncate=True,
        )
    
        # average other dims
        psd_iso_signal = psd_iso_signal.mean(dim=["time", "lat"], skipna=True)
        psd_iso_signal = psd_iso_signal.sel(freq_lon=psd_iso_signal["freq_lon"] > 0).compute()
    return psd_iso_signal


def isotropic_psd(da: xr.DataArray) -> xr.DataArray:
    with ProgressBar():
    
        # frequency dims
        psd_iso_signal = xrft.isotropic_power_spectrum(
            da.chunk({
                "time": 1,
                "lon": da.lon.shape[0], 
                "lat": da.lat.shape[0], }),
            dim=["lon", "lat"],
            detrend="linear",
            window="tukey",
            nfactor=2,
            window_correction=True,
            true_amplitude=True,
            truncate=True,
        )
    
        # average other dims
        psd_iso_signal = psd_iso_signal.mean(dim=["time", ], skipna=True)
        psd_iso_signal = psd_iso_signal.sel(freq_r=psd_iso_signal["freq_r"] > 0).compute()
    psd_iso_signal.name = da.name
    return psd_iso_signal

def space_time_psd(da: xr.DataArray) -> xr.DataArray:
    with ProgressBar():
    
        # frequency dims
        psd_iso_signal = xrft.power_spectrum(
            da.chunk({
                "time": da.time.shape[0],
                "lon": da.lon.shape[0], 
                "lat": 1,}),
            dim=["lon", "time",],
            detrend="linear",
            window="tukey",
            nfactor=2,
            window_correction=True,
            true_amplitude=True,
            truncate=True,
        )
        

    
        # average other dims
        psd_iso_signal = psd_iso_signal.mean(dim=["lat"], skipna=True)
        
        # drop frequency dims with negative frequenies
        psd_iso_signal = psd_iso_signal.sel(freq_lon=psd_iso_signal["freq_lon"] > 0)
        psd_iso_signal = psd_iso_signal.sel(freq_time=psd_iso_signal["freq_time"] > 0)
        
        psd_iso_signal = psd_iso_signal.compute()
        
    psd_iso_signal.name = da.name
    return psd_iso_signal




def psd_spacetime(da: xr.DataArray, dims: List[str], **kwargs) -> xr.DataArray:

    # compute PSD err and PSD signal
    psd_signal = xrft.power_spectrum(
        da,
        dim=dims,
        detrend=kwargs.get("detrend", "linear"),
        window=kwargs.get("window", "tukey"),
        nfactor=kwargs.get("nfactor", 2),
        window_correction=kwargs.get("window_correction", True),
        true_amplitude=kwargs.get("true_amplitude", True),
        truncate=kwargs.get("truncate", True),
    )

    return psd_signal


def psd_isotropic(da: xr.DataArray, dims: List[str], **kwargs) -> xr.DataArray:

    # compute PSD err and PSD signal
    psd_signal = xrft.isotropic_power_spectrum(
        da,
        dim=dims,
        detrend=kwargs.get("detrend", "linear"),
        window=kwargs.get("window", "tukey"),
        nfactor=kwargs.get("nfactor", 2),
        window_correction=kwargs.get("window_correction", True),
        true_amplitude=kwargs.get("true_amplitude", True),
        truncate=kwargs.get("truncate", True),
    )

    return psd_signal


class PlotPSDIsotropic:
    def init_fig(self, ax=None, figsize=None):
        if ax is None:
            figsize = (5, 4) if figsize is None else figsize
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.ax = ax
            self.fig = plt.gcf()

    def plot_wavenumber(self, da, freq_scale=1.0, units=None, **kwargs):

        if units is not None:
            xlabel = f"Wavenumber [cycles/{units}]"
        else:
            xlabel = f"Wavenumber"

        dim = list(da.dims)[0]

        self.ax.plot(da[dim] * freq_scale, da, **kwargs)

        self.ax.set(
            yscale="log",
            xscale="log",
            xlabel=xlabel,
            ylabel=f"PSD [{da.name}]",
            xlim=[10 ** (-3) - 0.00025, 10 ** (-1) + 0.025],
        )

        self.ax.legend()
        self.ax.grid(which="both", alpha=0.5)

    def plot_wavelength(self, da, freq_scale=1.0, units=None, **kwargs):

        if units is not None:
            xlabel = f"Wavelength [{units}]"
        else:
            xlabel = f"Wavelength"

        dim = list(da.dims)[0]

        self.ax.plot(1 / (da[dim] * freq_scale), da, **kwargs)

        self.ax.set(
            yscale="log", xscale="log", xlabel=xlabel, ylabel=f"PSD"
        )

        self.ax.xaxis.set_major_formatter("{x:.0f}")
        self.ax.invert_xaxis()

        self.ax.legend()
        self.ax.grid(which="both", alpha=0.5)

    def plot_both(self, da, freq_scale=1.0, units=None, **kwargs):

        if units is not None:
            xlabel = f"Wavelength [{units}]"
        else:
            xlabel = f"Wavelength"

        self.plot_wavenumber(da=da, units=units, freq_scale=freq_scale, **kwargs)

        self.secax = self.ax.secondary_xaxis(
            "top", functions=(lambda x: 1 / (x + 1e-20), lambda x: 1 / (x + 1e-20))
        )
        self.secax.xaxis.set_major_formatter("{x:.0f}")
        self.secax.set(xlabel=xlabel)


class PlotPSDSpaceTime:
    def init_fig(self, ax=None, figsize=None):
        if ax is None:
            figsize = (5,4) if figsize is None else figsize
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.ax = ax
            self.fig = plt.gcf()
        
    def plot_wavenumber(
        self, 
        da, 
        space_scale: float=1.0,
        space_units: str=None,
        time_units: str=None,
        psd_units: float=None,
        **kwargs):
        
        if space_units is not None:
            xlabel = f"Wavenumber [cycles/{space_units}]"
        else:
            xlabel = f"Wavenumber"
        if time_units is not None:
            ylabel = f"Frequency [cycles/{time_units}]"
        else:
            ylabel = f"Frequency"

        if psd_units is None:
            cbar_label = "PSD"
        else:
            cbar_label = f"PSD [{psd_units}]"
            
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        
        locator = ticker.LogLocator()
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        
        
        pts = self.ax.contourf(
            1/(da.freq_lon*space_scale),
            1/da.freq_time, 
            da.transpose("freq_time", "freq_lon"), 
            norm=norm, 
            locator=locator, 
            cmap=kwargs.pop("cmap", "RdYlGn"), 
            extend=kwargs.pop("extend", "both"),
            vmin=vmin, vmax=vmax,
            **kwargs
        )

        self.ax.set(
            yscale="log",
            xscale="log",
            xlabel=xlabel,
            ylabel=ylabel,
            
        )
        # colorbar
        fmt = ticker.LogFormatterMathtext(base=10)
        cbar = plt.colorbar(
            pts,
            ax=self.ax,
            pad=0.02,
            format=fmt,
            extend=True,
            norm=norm
            
        )
        cbar.ax.set_ylabel(cbar_label)
        self.ax.invert_xaxis()
        self.ax.invert_yaxis()
        self.ax.grid(which="both", linestyle="--", linewidth=1, color="black", alpha=0.2)

    def plot_wavelength(        
        self, 
        da, 
        space_scale: float=1.0,
        space_units: str=None,
        time_units: str=None,
        psd_units: float=None,
        **kwargs
    ):
    
        if space_units is not None:
            xlabel = f"Wavelength [{space_units}]"
        else:
            xlabel = f"Wavelength"
            
        if time_units is not None:
            ylabel = f"Period [{time_units}]"
        else:
            ylabel = f"Period"
            
        if psd_units is None:
            cbar_label = "PSD"
        else:
            cbar_label = f"PSD [{psd_units}]"
            
        self.plot_wavenumber(
            da=da, space_scale=space_scale, 
            space_units=space_units, time_units=time_units,
            psd_units=psd_units,
            **kwargs
        )

        self.ax.set(
            xlabel=xlabel, 
            ylabel=ylabel
        )
        self.ax.xaxis.set_major_formatter("{x:.0f}")
        self.ax.yaxis.set_major_formatter("{x:.0f}")


class PlotPSDFreqVariable:
    def init_fig(self, ax=None, figsize=None,):
        if ax is None:
            figsize = (5,4) if figsize is None else figsize
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.ax = ax
            self.fig = plt.gcf()
        
    def plot_wavenumber(
        self, 
        da, 
        variable: str,
        freq_scale: float=1.0,
        freq_units: str="m",
        var_units: str=None,
        psd_units: float=None,
        **kwargs):
        
        if freq_units is not None:
            xlabel = f"Wavenumber [cycles/{freq_units}]"
        else:
            xlabel = f"Wavenumber"
            
        var_name = da[variable].attrs.get("long_name", "")
        if var_units is not None:
            
            ylabel = var_name + f" [{var_units}]"
        else:
            ylabel = var_name

        if psd_units is None:
            cbar_label = "PSD"
        else:
            cbar_label = f"PSD [{psd_units}]"
            
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        
        locator = ticker.LogLocator()
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        
        
        pts = self.ax.contourf(
            1/(da.freq_lon*freq_scale),
            da[variable], 
            da.transpose(variable, "freq_lon"), 
            norm=norm, 
            locator=locator, 
            cmap=kwargs.pop("cmap", "RdYlGn"), 
            extend=kwargs.pop("extend", "both"),
            vmin=vmin, vmax=vmax,
            **kwargs
        )

        self.ax.set(
            xscale="log",
            xlabel=xlabel,
            ylabel=ylabel,
            
        )
        # colorbar
        fmt = ticker.LogFormatterMathtext(base=10)
        cbar = plt.colorbar(
            pts,
            ax=self.ax,
            pad=0.02,
            format=fmt,
            extend=True,
            norm=norm
            
        )
        cbar.ax.set_ylabel(cbar_label)
        self.ax.invert_xaxis()
        self.ax.grid(which="both", linestyle="--", linewidth=1, color="black", alpha=0.2)

    def plot_wavelength(        
        self, 
        da,
        variable: str,
        freq_scale: float=1.0,
        freq_units: str="m",
        var_units: str=None,
        psd_units: float=None,
        **kwargs):
    
        if freq_units is not None:
            xlabel = f"Wavelength [{freq_units}]"
        else:
            xlabel = f"Wavelength"
            
        var_name = da[variable].attrs.get("long_name", "")
        if var_units is not None:
            
            ylabel = var_name + f" [{var_units}]"
        else:
            ylabel = var_name

        if psd_units is None:
            cbar_label = "PSD"
        else:
            cbar_label = f"PSD [{psd_units}]"
            
        if psd_units is None:
            cbar_label = "PSD"
        else:
            cbar_label = f"PSD [{psd_units}]"
            
        self.plot_wavenumber(
            da=da, 
            variable=variable,
            freq_scale=freq_scale,
            freq_units=freq_units, 
            var_units=var_units,
            psd_units=psd_units,
            **kwargs
        )

        self.ax.set(
            xlabel=xlabel, 
            ylabel=ylabel
        )
        self.ax.xaxis.set_major_formatter("{x:.0f}")
        self.ax.yaxis.set_major_formatter("{x:.0f}")


