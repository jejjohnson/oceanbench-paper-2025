from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List

import cmocean
import copernicusmarine as cmc
import numpy as np
import xarray as xr


# Generate all Wednesdays in 2024
def generate_wednesdays_in_year(year: int = 2024):
    d = datetime(int(year), 1, 1)
    d += timedelta(days=(2 - d.weekday()) % 7)  # First Wednesday
    while d.year == year:
        yield d
        d += timedelta(days=7)


@dataclass
class Model:
    name: str
    color: str
    linestyle: str
    forecast: str
    diagnostic: str
    trajectories: str

    def forecast_path(self, year: str = "20240103"):
        return f"{self.forecast}{year}.zarr"

    def diagnostic_path(self, year: str = "20240103"):
        return f"{self.diagnostic}{year}.zarr"

    def trajectories_path(self, year: str = "20240103"):
        return f"{self.trajectories}{year}.zarr"


@dataclass
class GLONET(Model):
    name: str = "glonet"
    forecast: str = (
        "https://minio.dive.edito.eu/project-oceanbench/public/glonet_full_2024/"
    )
    diagnostic: str = (
        "https://minio.dive.edito.eu/project-oceanbench/public/tmp/GLONET/process/"
    )
    trajectories: str = (
        "https://minio.dive.edito.eu/project-oceanbench/public/tmp/GLONET/trajs/"
    )
    color: str = "blue"
    linestyle: str = "--"


@dataclass
class GLO12(Model):
    name: str = "glo12"
    forecast: str = "https://minio.dive.edito.eu/project-oceanbench/public/GLO12/"
    diagnostic: str = (
        "https://minio.dive.edito.eu/project-oceanbench/public/tmp/GLO12/process/"
    )
    trajectories: str = (
        "https://minio.dive.edito.eu/project-oceanbench/public/tmp/GLO12/trajs/"
    )
    color: str = "red"
    linestyle: str = "-"


@dataclass
class WENHAI(Model):
    name: str = "wenhai"
    forecast: str = "https://minio.dive.edito.eu/project-oceanbench/public/WENHAI/"
    diagnostic: str = (
        "https://minio.dive.edito.eu/project-oceanbench/public/tmp/WENHAI/process/"
    )
    trajectories: str = (
        "https://minio.dive.edito.eu/project-oceanbench/public/tmp/WENHAI/trajs/"
    )
    color: str = "orange"
    linestyle: str = ".-"


@dataclass
class XIHE(Model):
    name: str = "xihe"
    forecast: str = "https://minio.dive.edito.eu/project-oceanbench/public/XIHE/"
    diagnostic: str = (
        "https://minio.dive.edito.eu/project-oceanbench/public/tmp/XIHE/process/"
    )
    trajectories: str = (
        "https://minio.dive.edito.eu/project-oceanbench/public/tmp/XIHE/trajs/"
    )
    color: str = "green"
    linestyle: str = "--"


MODELS = {"glonet": GLONET(), "glo12": GLO12(), "wenhai": WENHAI(), "xihe": XIHE()}


@dataclass
class Variable:
    name: str
    standard_name: str
    long_name: str
    units: str
    cmap: str
    levels: int
    linestyles: str | Callable

    def correct_real_attrs(self, da: xr.Dataset | xr.DataArray):
        da.attrs["units"] = self.units
        da.attrs["standard_name"] = self.standard_name
        da.attrs["long_name"] = self.long_name
        return da

    def correct_spectral_attrs(self, da: xr.Dataset | xr.DataArray):
        da.attrs["units"] = f"1/{self.units}"
        da.attrs["standard_name"] = self.standard_name
        da.attrs["long_name"] = self.long_name
        return da


@dataclass
class SeaSurfaceHeight(Variable):
    name: str = "zos"
    standard_name: str = "sea_surface_height"
    long_name: str = "Sea Surface Height"
    units: str = "m"
    cmap: str = "viridis"
    levels: int = 5
    linestyles: str | Callable = lambda levels: np.where(levels >= 0, "-", "--")


@dataclass
class Temperature(Variable):
    name: str = "thetao"
    standard_name: str = "temperature"
    long_name: str = "Sea Water Potential Temperature"
    units: str = "°C"
    cmap: str = "RdBu_r"
    levels: int = 5
    linestyles: str = "-"


@dataclass
class Salinity(Variable):
    name: str = "so"
    standard_name: str = "sea_water_salinity"
    long_name: str = "Sea Water Salinity"
    units: str = "PSU"
    cmap: str = field(default_factory=lambda: cmocean.cm.haline)  # "YlGnBu_r"
    levels: int = 5
    linestyles: str = "-"


@dataclass
class MixedLayerDepth(Variable):
    name: str = "MLD"
    standard_name: str = "mixed_layer_depth"
    long_name: str = "Mixed Layer Depth"
    units: str = "m"
    cmap: str = field(default_factory=lambda: cmocean.cm.deep)  # "YlGnBu_r"
    levels: int = 5
    linestyles: str = "-"


@dataclass
class ZonalVelocity(Variable):
    name: str = "uo"
    standard_name: str = "zonal_current"
    long_name: str = "Zonal Current"
    units: str = "m/s"
    cmap: str = field(default_factory=lambda: cmocean.cm.speed)  # "YlGnBu_r"
    levels: int = 5
    linestyles: str = "-"


@dataclass
class MeridionalVelocity(ZonalVelocity):
    name: str = "vo"
    standard_name: str = "meridional_current"
    long_name: str = "Meridional Current"


@dataclass
class GeostrophicZonalVelocity(ZonalVelocity):
    name: str = "u_geo"
    standard_name: str = "geostrophic_eastward_sea_water_velocity"
    long_name: str = "Geostrophic Zonal Velocity"


@dataclass
class GeostrophicMeridionalVelocity(ZonalVelocity):
    name: str = "v_geo"
    standard_name: str = "geostrophic_northward_sea_water_velocity"
    long_name: str = "Geostrophic Meridional Velocity"


FORECAST_VARIABLES = {
    "zos": SeaSurfaceHeight(),
    "thetao": Temperature(),
    "uo": ZonalVelocity(),
    "vo": MeridionalVelocity(),
    "so": Salinity(),
}
DIAGNOSTIC_VARIABLES = {
    "MLD": MixedLayerDepth(),
    "u_geo": GeostrophicZonalVelocity(),
    "v_geo": GeostrophicMeridionalVelocity(),
}
TRAJECTORY_VARIABLES = ["lat", "lon"]


@dataclass
class ForecastDataset:
    models: Dict[str, Model] = field(default_factory=lambda: MODELS)
    variables: List[Variable] = field(default_factory=lambda: FORECAST_VARIABLES)


@dataclass
class DiagnosticDataset:
    models: Dict[str, Model] = field(default_factory=lambda: MODELS)
    variables: List[Variable] = field(default_factory=lambda: DIAGNOSTIC_VARIABLES)


@dataclass
class TrajectoriesDataset:
    models: Dict[str, Model] = field(default_factory=lambda: MODELS)
    variables: List[Variable] = field(default_factory=lambda: TRAJECTORY_VARIABLES)


from datetime import datetime

import copernicusmarine as cmc


@dataclass
class ReAnalysis:
    name: str = "glorys12"
    color: str = "black"
    linestyle: str = "-"

    def load_data(
        self,
        variable: str = "zos",
        date: datetime = datetime(2020, 1, 4),
        depth: float = 0.49,
    ):
        depth_min, depth_max = depth - 0.01, depth + 0.01
        data = cmc.open_dataset(
            dataset_id="cmems_mod_glo_phy_myint_0.083deg_P1D-m",  # "cmems_mod_glo_phy_my_0.083deg_P1D-m",
            variables=[variable],
            minimum_longitude=-180,
            maximum_longitude=180,
            minimum_latitude=-80,
            maximum_latitude=90,
            minimum_depth=depth_min,
            maximum_depth=depth_max,
            start_datetime=date,
            end_datetime=date,
        )
        return data
