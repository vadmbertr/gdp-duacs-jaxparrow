import clouddrift as cd
import numpy as np
import xarray as xr


def gps_only(ds: xr.Dataset) -> xr.Dataset:
    return cd.ragged.subset(ds, {"location_type": True}, row_dim_name="traj")  # True means GPS / False Argos


def after_2000_only(ds: xr.Dataset) -> xr.Dataset:
    return cd.ragged.subset(
        ds,
        {"deploy_date": lambda dt: dt >= np.datetime64("2000-01-01")},
        row_dim_name="traj"
    )


def svp_only(ds: xr.Dataset) -> xr.Dataset:
    return cd.ragged.subset(
        ds,
        {"typebuoy": lambda tb: np.char.find(tb.astype(str), "SVP") != -1},
        row_dim_name="traj"
    )


def before_2023_06_07_only(ds: xr.Dataset) -> xr.Dataset:
    return cd.ragged.subset(ds, {"time": lambda dt: dt < np.datetime64("2023-06-07")}, row_dim_name="traj")


def drogued_only(ds: xr.Dataset) -> xr.Dataset:
    return cd.ragged.subset(ds, {"drogue_status": True}, row_dim_name="traj")


def remove_low_latitudes(ds: xr.Dataset, cutoff: float = 5) -> xr.Dataset:
    return cd.ragged.subset(ds, {"lat": lambda arr: np.abs(arr) > cutoff}, row_dim_name="traj")


def finite_value_only(ds: xr.Dataset) -> xr.Dataset:
    ds = cd.ragged.subset(ds, {"lat": np.isfinite}, row_dim_name="traj")
    ds = cd.ragged.subset(ds, {"lon": np.isfinite}, row_dim_name="traj")
    ds = cd.ragged.subset(ds, {"vn": np.isfinite}, row_dim_name="traj")
    ds = cd.ragged.subset(ds, {"ve": np.isfinite}, row_dim_name="traj")
    ds = cd.ragged.subset(ds, {"time": lambda arr: ~np.isnat(arr)}, row_dim_name="traj")
    return ds


def remove_outlier_values(ds: xr.Dataset, cutoff: float = 10) -> xr.Dataset:
    def velocity_cutoff(arr: xr.DataArray) -> xr.DataArray:
        return np.abs(arr) <= cutoff

    ds = cd.ragged.subset(ds, {"vn": velocity_cutoff}, row_dim_name="traj")
    ds = cd.ragged.subset(ds, {"ve": velocity_cutoff}, row_dim_name="traj")
    return ds
