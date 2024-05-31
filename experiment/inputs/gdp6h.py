from collections.abc import Callable

import clouddrift as cd
import xarray as xr

from ..filesystem.zarr_store import ZarrStore


class GDP6hDs:
    def __init__(
        self,
        gdp6h_url: str = "https://www.aoml.noaa.gov/ftp/pub/phod/buoydata/gdp6h_ragged_may23.nc"
    ):
        self.gdp6h_url = gdp6h_url
        self.dataset = None
        self.filter_fn = lambda x: x
        self.name = "global"

    def load_data(self):
        ds = cd.datasets.gdp6h()
        ds = self.filter_fn(ds)

        self.dataset = ds

    def apply_preproc(self, preproc_fn: Callable):
        self.dataset = preproc_fn(self.dataset)
        self.select_variables()
        self.reset_chunk()

    def select_variables(self):
        variables = [  # only keep relevant variables
            "rowsize",
            "lon",
            "lat",
            "time",
            "ve",
            "vn"
        ]
        self.dataset = self.dataset[variables]

    def reset_chunk(self):
        self.dataset = self.dataset.chunk(chunks="auto")

    def dump_data(self, store: ZarrStore):
        self.dataset.to_zarr(store.store, mode="w")


def mediterranean_masking(ds: xr.Dataset) -> xr.Dataset:
    def compute_mask(lon: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
        # Mediterranean sea (-6.0327, 36.2173, 30.2639, 45.7833)
        in_mediterranea = (-6.0327 <= lon) & (lon <= 36.2173) & (30.2639 <= lat) & (lat <= 45.7833)

        # Bay of Biscay
        in_biscay = (lon <= -0.1462) & (lat >= 43.2744)
        # Black Sea
        in_blacksea = (lon >= 27.4437) & (lat >= 40.9088)
        # exclude them
        mask = in_mediterranea & ~(in_biscay | in_blacksea)

        return mask

    ds = cd.ragged.subset(ds, {("lon", "lat"): compute_mask}, row_dim_name="traj")

    return ds


def alboran_masking(ds: xr.Dataset) -> xr.Dataset:
    def compute_mask(lon: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
        # Alboran sea (-5.3538, -1.1883, 35.0707, 36.8415)
        in_alboran = (-5.3538 <= lon) & (lon <= -1.1883) & (35.0707 <= lat) & (lat <= 36.8415)
        return in_alboran

    ds = cd.ragged.subset(ds, {("lon", "lat"): compute_mask}, row_dim_name="traj")

    return ds
