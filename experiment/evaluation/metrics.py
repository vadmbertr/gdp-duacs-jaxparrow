import numpy as np
import pandas as pd
from jaxtyping import Array, Float
import vaex
import xarray as xr


def _euclidean_dist(
        u: xr.DataArray, u_hat: xr.DataArray,
        v: xr.DataArray, v_hat: xr.DataArray
) -> (xr.DataArray, xr.DataArray, xr.DataArray):
    err_u = (u_hat - u)**2
    err_v = (v_hat - v)**2
    err_uv = (err_u + err_v)**.5
    err_u **= .5
    err_v **= .5
    return err_u, err_v, err_uv


def compute_along_traj_metrics(gdp6h_ds: xr.Dataset, methods: []) -> xr.Dataset:
    traj_metrics = gdp6h_ds
    for method in methods:
        traj_metrics[f"err_u_{method}"], traj_metrics[f"err_v_{method}"], traj_metrics[f"err_{method}"] = (
            _euclidean_dist(gdp6h_ds.ve, gdp6h_ds[f"u_hat_{method}"], gdp6h_ds.vn, gdp6h_ds[f"v_hat_{method}"])
        )
        traj_metrics = traj_metrics.drop_vars([f"u_hat_{method}", f"v_hat_{method}"])   # we don't need those anymore

    return traj_metrics.drop_vars(["ve", "vn"])


def _add_fake_rows(duacs_ds: xr.Dataset, traj_metrics: pd.DataFrame) -> pd.DataFrame:
    hdlat = (duacs_ds.latitude[1] - duacs_ds.latitude[0]) / 2
    hdlon = (duacs_ds.longitude[1] - duacs_ds.longitude[0]) / 2

    fake_data = [np.nan] * (len(traj_metrics.columns) - 2)  # - 2 because with fill 2 columns: lon & lat
    fake_min_data = [duacs_ds.longitude.min().data - hdlon, duacs_ds.latitude.min().data - hdlat] + fake_data
    fake_max_data = [duacs_ds.longitude.max().data + hdlon, duacs_ds.latitude.max().data + hdlat] + fake_data

    fake_min_row = pd.Series(fake_min_data, index=traj_metrics.columns)
    fake_max_row = pd.Series(fake_max_data, index=traj_metrics.columns)
    fake_rows = pd.concat([fake_min_row, fake_max_row], axis=1).T
    fake_rows = fake_rows.astype({k: t for k, t in zip(traj_metrics.columns, traj_metrics.dtypes.tolist())})

    return pd.concat([traj_metrics, fake_rows], ignore_index=True)


def compute_binned_metrics(
        duacs_ds: xr.Dataset,
        traj_metrics: xr.Dataset,
        methods: []
) -> [Float[Array, "time lat lon"], ...]:
    binby = ["lat", "lon"]
    sum_expression = ([f"err_u_{method}" for method in methods] +
                      [f"err_v_{method}" for method in methods] +
                      [f"err_{method}" for method in methods])
    count_expression = ([f"isfinite(err_u_{method})" for method in methods] +
                        [f"isfinite(err_v_{method})" for method in methods] +
                        [f"isfinite(err_{method})" for method in methods])
    domain_shape = [duacs_ds.latitude.n_locations, duacs_ds.longitude.n_locations]

    # convert to dataframe
    traj_metrics = traj_metrics.to_dataframe().reset_index().drop("obs", axis=1)
    # add fake rows ensuring the binning results in the original domain
    traj_metrics = _add_fake_rows(duacs_ds, traj_metrics)

    # to vaex dataframe
    traj_metrics = vaex.from_pandas(traj_metrics)
    # binning
    # sum (mean/var are computed afterward to account for the potential batching)
    binned_sums = traj_metrics.sum(
        sum_expression,
        binby=binby,
        shape=domain_shape
    )
    # count
    binned_count = traj_metrics.sum(
        count_expression,
        binby=binby,
        shape=domain_shape
    )

    return binned_sums, binned_count, sum_expression
