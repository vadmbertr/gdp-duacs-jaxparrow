import clouddrift as cd
import numpy as np
import xarray as xr


def gps_only(ds: xr.Dataset) -> xr.Dataset:
    ds.location_type.load()
    return cd.ragged.subset(ds, {"location_type": True})  # True means GPS / False Argos


def after_2000_only(ds: xr.Dataset) -> xr.Dataset:
    ds.deploy_date.load()
    return cd.ragged.subset(ds, {"deploy_date": lambda dt: dt >= np.datetime64("2000-01-01")})


def svp_only(ds: xr.Dataset) -> xr.Dataset:
    ds.typebuoy.load()
    return cd.ragged.subset(ds, {"typebuoy": lambda tb: np.char.find(tb.astype(str), "SVP") != -1})


def before_2023_06_07_only(ds: xr.Dataset) -> xr.Dataset:
    ds.deploy_date.load()
    return cd.ragged.subset(ds, {"time": lambda dt: dt < np.datetime64("2023-06-07")})


def drogued_only(ds: xr.Dataset) -> xr.Dataset:
    ds.drogue_status.load()
    return cd.ragged.subset(ds, {"drogue_status": True})


def remove_low_latitudes(ds: xr.Dataset, cutoff: float = 5) -> xr.Dataset:
    ds.lat.load()
    return cd.ragged.subset(ds, {"lat": lambda arr: np.abs(arr) > cutoff})


def finite_value_only(ds: xr.Dataset) -> xr.Dataset:
    ds.lat.load()
    ds = cd.ragged.subset(ds, {"lat": np.isfinite})
    ds.lon.load()
    ds = cd.ragged.subset(ds, {"lon": np.isfinite})
    ds.vn.load()
    ds = cd.ragged.subset(ds, {"vn": np.isfinite})
    ds.ve.load()
    ds = cd.ragged.subset(ds, {"ve": np.isfinite})
    ds.time.load()
    ds = cd.ragged.subset(ds, {"time": lambda arr: ~np.isnat(arr)})
    return ds


def remove_outlier_values(ds: xr.Dataset, cutoff: float = 10) -> xr.Dataset:
    def velocity_cutoff(arr: xr.DataArray) -> xr.DataArray:
        return np.abs(arr) <= cutoff

    ds.vn.load()
    ds = cd.ragged.subset(ds, {"vn": velocity_cutoff})
    ds.ve.load()
    ds = cd.ragged.subset(ds, {"ve": velocity_cutoff})
    return ds
