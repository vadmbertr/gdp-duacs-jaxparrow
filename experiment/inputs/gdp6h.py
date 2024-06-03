from collections.abc import Callable
import io

import clouddrift as cd
import numpy as np
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

    def load_data(self, store: ZarrStore):
        def sanitize_date(arr: xr.DataArray) -> xr.DataArray:
            nat_index = arr < 0
            arr[nat_index.compute()] = np.nan
            return arr

        # the GDP6H inputs from clouddrift can not be opened "remotely", so we store it (for now)
        # (see https://github.com/Cloud-Drift/clouddrift/issues/363)
        if not store.exists():  # download inputs: takes time
            # from cd.datasets.gdp6h
            buffer = io.BytesIO()
            cd.adapters.utils.download_with_progress([(f"{self.gdp6h_url}#mode=bytes", buffer, None)])
            reader = io.BufferedReader(buffer)
            ds = xr.open_dataset(reader)
            ds = ds.rename_vars({"ID": "id"}).assign_coords({"id": ds.ID}).drop_vars(["ids"])
            ds.to_zarr(store.store, mode="w")

        ds = xr.open_zarr(store.store, decode_times=False)
        ds["deploy_date"] = sanitize_date(ds.deploy_date)
        ds["end_date"] = sanitize_date(ds.end_date)
        ds["drogue_lost_date"] = sanitize_date(ds.drogue_lost_date)
        ds["time"] = sanitize_date(ds.time)
        ds = xr.decode_cf(ds)

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

    ds.lon.load()
    ds.lat.load()
    ds = cd.ragged.subset(ds, {("lon", "lat"): compute_mask}, row_dim_name="traj")

    return ds


def alboran_masking(ds: xr.Dataset) -> xr.Dataset:
    def compute_mask(lon: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
        # Alboran sea (-5.3538, -1.1883, 35.0707, 36.8415)
        in_alboran = (-5.3538 <= lon) & (lon <= -1.1883) & (35.0707 <= lat) & (lat <= 36.8415)
        return in_alboran

    ds = cd.ragged.subset(ds, {("lon", "lat"): compute_mask}, row_dim_name="traj")

    return ds
