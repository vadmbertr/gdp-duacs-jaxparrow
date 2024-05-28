import os

import cachier
import copernicusmarine as cm
import numpy as np

from ..logger.logger import LOGGER
from .gdp6h import GDP6hDs


class DuacsDs:
    def __init__(
            self,
            cms_username_env_var: str = "COPERNICUS_MARINE_SERVICE_USERNAME",
            cms_password_env_var: str = "COPERNICUS_MARINE_SERVICE_PASSWORD",
            cms_dataset_id: str = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D",
            disable_caching: bool = True
    ):
        if disable_caching:
            cachier.disable_caching()

        cm.login(os.environ[cms_username_env_var], os.environ[cms_password_env_var], overwrite_configuration_file=True)
        self.dataset_id = cms_dataset_id
        self.dataset = None
        self.name = ""

    def load_data(self, gdp6_ds: GDP6hDs):
        drifters_lon = gdp6_ds.dataset.lon.values
        drifters_lat = gdp6_ds.dataset.lat.values
        min_lon = max(-179.875, drifters_lon.min() - .5)  # add inputs spatial resolution
        max_lon = min(179.875, drifters_lon.max() + .5)
        min_lat = max(-89.875, drifters_lat.min() - .5)
        max_lat = min(89.875, drifters_lat.max() + .5)
        spatial_extent = (min_lon, max_lon, min_lat, max_lat)

        drifters_time = gdp6_ds.dataset.time.values
        temporal_slice = (drifters_time.min() - np.timedelta64(1, "D"),
                          drifters_time.max() + np.timedelta64(2, "D"))  # add inputs temporal resolution

        variables = ["adt", "ugos", "vgos"]  # we retrieve SSH data and U V data (for comparison)

        dataset_options = {
            "dataset_id": self.dataset_id,
            "dataset_version": "202112",
            "dataset_part": "default",
            "service": "arco-time-series",
            "variables": variables,
            "minimum_longitude": spatial_extent[0],
            "maximum_longitude": spatial_extent[1],
            "minimum_latitude": spatial_extent[2],
            "maximum_latitude": spatial_extent[3],
            "start_datetime": temporal_slice[0].astype("datetime64[s]"),
            "end_datetime": temporal_slice[1].astype("datetime64[s]")
        }

        ds = None
        while ds is None:
            try:  # fetching the catalog might fail
                ds = cm.open_dataset(**dataset_options)
            except Exception as e:
                LOGGER.warning(f"catch exception {e}")

        # restrict to times of interest
        drifters_time = np.unique(drifters_time.astype("datetime64[D]"))
        drifters_time_prev = drifters_time - np.timedelta64(1, "D")
        drifters_time_next = drifters_time + np.timedelta64(1, "D")
        times = np.unique(np.concatenate([drifters_time, drifters_time_prev, drifters_time_next]))
        ds = ds.sel(time=ds.time.isin(times))

        self.dataset = ds
        self.name = gdp6_ds.name
