import xarray as xr
from typing import Callable
import cmocean
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
import matplotlib.pyplot as plt


@dataclass
class PlotterContour:
    da: xr.DataArray
    config: object

    def __post_init__(self):
        self.correct_labels()

    def correct_labels(self):
        self.da["lon"].attrs["units"] = "degrees"
        self.da["lat"].attrs["units"] = "degrees"
        self.da.attrs["units"] = self.config.units
        self.da.attrs["standard_name"] = self.config.standard_name
        self.da.attrs["long_name"] = self.config.long_name
        self.da["lon"].attrs["standard_name"] = "longitude"
        self.da["lat"].attrs["standard_name"] = "latitude"
        self.da["lat"].attrs["long_name"] = "Latitude"
        self.da["lon"].attrs["long_name"] = "Longitude"

    def plot_figure(self, **kwargs):

        fig, ax = plt.subplots(figsize=(8,7), subplot_kw={'projection': ccrs.PlateCarree()})
        vmin=kwargs.pop("vmin", self.da.min().values)
        vmax=kwargs.pop("vmax", self.da.max().values)
        cmap=kwargs.pop("cmap", self.config.cmap)
        levels = kwargs.pop("levels", self.config.levels)
        levels = levels if levels else None
        
        self.da.plot.pcolormesh(
            ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
            transform=ccrs.PlateCarree(),
            cbar_kwargs=kwargs.pop("cbar_kwargs", None),
            **kwargs,
        )
        if levels:
            loc = ticker.MaxNLocator(levels)
            levels = loc.tick_values(self.da.min().values, self.da.max().values)
            self.da.plot.contour(
                ax=ax, 
                alpha=0.5, linewidths=1, cmap="black",
                levels=levels,
                # linestyles=self.config.linestyles
                # vmin=vmin, vmax=vmax,
                # **kwargs
            )    
    
    
        ax.coastlines(linewidth=1)
        # ax.set(title=)
        
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.1, color='k', alpha=1, 
                          linestyle='--')
        
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12} 
    
        # Add map features with Cartopy 
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', 
                                                    # edgecolor='face', 
                                                    facecolor='lightgray'))
        
        # ax.set_title(pd.to_datetime(self.da.time.values).strftime('%Y-%m-%d'))
        ax.set_title("")
        fig.tight_layout()
        fig.set(dpi=300)
        
        return fig, ax

# def plot_ssh_map(ds, variable: str="ssh", **kwargs):
    
#     fig, ax = plt.subplots(figsize=(7,5.5))
#     vmin=kwargs.pop("vmin", ds[variable].min().values)
#     vmax=kwargs.pop("vmax", ds[variable].max().values)
#     cmap=kwargs.pop("cmap", "viridis")
#     levels = kwargs.pop("levels", 5)
    
#     ds[variable].plot.pcolormesh(
#         ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
#         cbar_kwargs=kwargs.pop("cbar_kwargs", None),
#         **kwargs,
#     )
#     loc = ticker.MaxNLocator(levels)
#     levels = loc.tick_values(ds[variable].min().values, ds[variable].max().values)
#     ds[variable].plot.contour(
#         ax=ax, 
#         alpha=0.5, linewidths=1, cmap="black",
#         levels=levels,
#         linestyles=np.where(levels >= 0, "-", "--")
#         # vmin=vmin, vmax=vmax,
#         # **kwargs
#     )    
#     ax.set_title(pd.to_datetime(ds.time.values).strftime('%Y-%m-%d'))
#     fig.tight_layout()
    
#     return fig, ax