import math

import clouddrift as cd
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
import xarray as xr

from ..inputs.duacs import DuacsDs, GDP6hDs
from ..evaluation.comparison import compare_methods
from ..evaluation.interpolation import interpolate_grid, interpolate_drifters_location
from ..evaluation.kinematics import compute_kinematics
from ..evaluation.metrics import compute_along_traj_metrics, compute_binned_metrics
from ..logger.logger import LOGGER


def _estimate_batch_indices(n_time: int, n_lat: int, n_lon: int, memory_per_device: float) -> list:
    f32_size = 4  # mostly manipulate f32 arrays
    comp_mem_per_time = f32_size * n_lat * n_lon * 1e-9  # in Gb
    comp_mem_per_time *= 40  # empirical factor preventing OOM errors
    batch_size = int(memory_per_device // comp_mem_per_time)  # conservative batch size
    n_batches = math.ceil(n_time / batch_size)  # conservative number of batches
    indices = jnp.arange(1, n_batches) * batch_size
    return indices.tolist()


def _normalize_datasets(ssc_fields_ds: xr.Dataset, methods: []) -> xr.Dataset:
    def _manual_mean(dataset: xr.Dataset, var: str) -> np.ndarray:
        return (dataset[f"{var}_sum"] / dataset[f"{var}_count"]).data

    data_vars = {}
    for method in methods:
        u_avg = _manual_mean(ssc_fields_ds, f"u_{method}")
        data_vars[f"u_avg_{method}"] = (
            ["latitude", "longitude"],
            u_avg,
            {"method": method, "what": "$\\langle \\mathbf{\\hat{u}}_u \\rangle$", "units": "$m/s$"}
        )
        v_avg = _manual_mean(ssc_fields_ds, f"v_{method}")
        data_vars[f"v_avg_{method}"] = (
            ["latitude", "longitude"],
            v_avg,
            {"method": method, "what": "$\\langle \\mathbf{\\hat{u}}_v \\| \\rangle$", "units": "$m/s$"}
        )
        data_vars[f"magn_avg_{method}"] = (
            ["latitude", "longitude"],
            _manual_mean(ssc_fields_ds, f"magn_{method}"),
            {"method": method, "what": "$\\langle \\| \\mathbf{\\hat{u}} \\| \\rangle$", "units": "$m/s$"}
        )
        data_vars[f"nrv_avg_{method}"] = (
            ["latitude", "longitude"],
            _manual_mean(ssc_fields_ds, f"nrv_{method}"),
            {"method": method, "what": "$\\langle \\xi/f \\rangle$", "units": ""}
        )
        u_sq_avg = _manual_mean(ssc_fields_ds, f"u_sq_{method}")
        v_sq_avg = _manual_mean(ssc_fields_ds, f"v_sq_{method}")
        data_vars[f"eke_avg_{method}"] = (
            ["latitude", "longitude"],
            (u_sq_avg - u_avg**2 + v_sq_avg - v_sq_avg**2) / 2,
            {"method": method, "what": "$\\langle \\text{EKE} \\rangle$", "units": "$m^2/s^{-2}$"}
        )
        data_vars[f"err_u_avg_{method}"] = (
            ["latitude", "longitude"],
            _manual_mean(ssc_fields_ds, f"err_u_{method}"),
            {"method": method, "what": "$\\langle \\| \\mathbf{u}_u - \\mathbf{\\hat{u}}_u \\| \\rangle$",
             "units": "$m/s$"}
        )
        data_vars[f"err_v_avg_{method}"] = (
            ["latitude", "longitude"],
            _manual_mean(ssc_fields_ds, f"err_v_{method}"),
            {"method": method, "what": "$\\langle \\| \\mathbf{u}_v - \\mathbf{\\hat{u}}_v \\| \\rangle$",
             "units": "$m/s$"}
        )
        data_vars[f"err_avg_{method}"] = (
            ["latitude", "longitude"],
            _manual_mean(ssc_fields_ds, f"err_{method}"),
            {"method": method, "what": "$\\langle \\| \\mathbf{u} - \\mathbf{\\hat{u}} \\| \\rangle$", "units": "$m/s$"}
        )
    return xr.Dataset(data_vars=data_vars, coords=ssc_fields_ds.coords)


def _add_ssh(
        ssc_fields_ds: xr.Dataset,
        duacs_ds: xr.Dataset,
) -> xr.Dataset:
    ssc_fields_ds["adt"] = duacs_ds.adt.mean(dim="time")
    ssc_fields_ds.adt.attrs = {"method": "DUACS", "what": duacs_ds.adt.attrs["long_name"], "units": "$m$"}
    return ssc_fields_ds


def estimate_and_evaluate(
        gdp6h_ds: GDP6hDs,
        duacs_ds: DuacsDs,
        methods: dict,
        memory_per_device: int
) -> xr.Dataset:
    duacs_xr = duacs_ds.dataset

    # fix over batches
    lat_t = jnp.ones((duacs_xr.latitude.size, duacs_xr.longitude.size)) * duacs_xr.latitude.data.reshape(-1, 1)
    lon_t = jnp.ones((duacs_xr.latitude.size, duacs_xr.longitude.size)) * duacs_xr.longitude.data

    # estimate batch indices based on domain dimensions
    n_time, n_lat, n_lon = tuple(duacs_xr.sizes.values())
    batch_indices = _estimate_batch_indices(n_time, n_lat, n_lon, memory_per_device)
    batch_indices = [0] + batch_indices + [n_time]

    # apply per batch
    ssc_fields_ds, methods_batch = None, None
    for idx0, idx1 in zip(batch_indices[:-1], batch_indices[1:]):
        LOGGER.debug(f"2.0.i. mini-batch {[idx0, idx1]}")
        ssc_fields_ds_batch, methods_batch = process_batch(
            idx0, idx1, gdp6h_ds.dataset, duacs_xr, lat_t, lon_t, methods
        )
        if ssc_fields_ds is None:
            ssc_fields_ds = ssc_fields_ds_batch
        else:
            ssc_fields_ds += ssc_fields_ds_batch

    # get spatial mask
    mask = ssc_fields_ds.mask

    LOGGER.info("2.5. Normalizing batched metrics")
    ssc_fields_ds = _normalize_datasets(ssc_fields_ds, methods_batch)

    LOGGER.info("2.6. Comparing methods")
    ssc_fields_ds = compare_methods(ssc_fields_ds, methods.keys())

    # logger.info("2.7. Adding mean SSH")
    # ssc_fields_ds = _add_ssh(ssc_fields_ds, duacs_xr)

    LOGGER.info("2.8. Applying spatial mask")
    ssc_fields_ds = ssc_fields_ds.where(mask)

    return ssc_fields_ds


def _kinematics_to_dataset(
        kinematics_vars: dict,
        lat_t: Float[Array, "lat lon"],
        lon_t: Float[Array, "lat lon"]
) -> xr.Dataset:
    uv_coords = {
        "latitude": (["latitude"], np.unique(lat_t).astype(np.float32)),
        "longitude": (["longitude"], np.unique(lon_t).astype(np.float32))
    }
    data_vars = {}
    for key, field in kinematics_vars.items():
        data_vars[f"{key}_sum"] = (["latitude", "longitude"], np.nansum(field, axis=0))
        data_vars[f"{key}_count"] = (["latitude", "longitude"], np.isfinite(field).sum(axis=0))

    return xr.Dataset(data_vars=data_vars, coords=uv_coords)


def _err_to_dataset(
        ssc_fields_ds: xr.Dataset,
        sum_expression: [],
        binned_sums: [Float[Array, "lat lon"], ...],
        binned_counts: [Float[Array, "lat lon"], ...]
) -> xr.Dataset:
    data_vars = {}
    for expr, binned_sum, binned_count in zip(sum_expression, binned_sums, binned_counts):
        data_vars[f"{expr}_sum"] = (["latitude", "longitude"], binned_sum)
        data_vars[f"{expr}_count"] = (["latitude", "longitude"], binned_count)

    return ssc_fields_ds.assign(**data_vars)


def _mask_to_dataset(ssc_fields_ds: xr.Dataset, mask: Float[Array, "time lat lon"]) -> xr.Dataset:
    ssc_fields_ds["mask"] = (["latitude", "longitude"], ~(mask.all(axis=0)))
    return ssc_fields_ds


def process_batch(
        idx0: int,
        idx1: int,
        gdp6h_ds: xr.Dataset,
        duacs_ds: xr.Dataset,
        lat_t: Float[Array, "lat lon"],
        lon_t: Float[Array, "lat lon"],
        methods: dict
) -> (xr.Dataset, []):
    LOGGER.info("2.1. Estimating SSC")
    adt_t = duacs_ds.adt.isel(time=slice(idx0, idx1)).data
    mask = ~np.isfinite(adt_t)
    u, v, lat_u, lon_u, lat_v, lon_v = None, None, None, None, None, None
    uv_fields = {}
    for method, f in methods.items():
        LOGGER.debug(f"2.1.i. Method: {method}")
        del u, v, lat_u, lon_u, lat_v, lon_v  # make sure we do not accumulate arrays in gpu memory
        u, v, lat_u, lon_u, lat_v, lon_v = f(lat_t, lon_t, adt_t, mask)
        u = jnp.where(jnp.abs(u) <= 10, u, jnp.nan)  # nan incoherent velocities (abs larger than 10m/s)
        v = jnp.where(jnp.abs(v) <= 10, v, jnp.nan)
        uv_fields[method] = (np.array(u, dtype=np.float32), np.array(v, dtype=np.float32))  # to cpu

    # add DUACS velocities (interpolated to the U and V points)
    u = interpolate_grid(duacs_ds.ugos.isel(time=slice(idx0, idx1)).data, axis=1, padding="right")
    v = interpolate_grid(duacs_ds.vgos.isel(time=slice(idx0, idx1)).data, axis=0, padding="right")
    uv_fields["DUACS"] = (np.array(u, dtype=np.float32), np.array(v, dtype=np.float32))

    LOGGER.info("2.2. Interpolating SSC velocities to drifters positions")
    time = duacs_ds.time[idx0:idx1].data
    gdp6h_ds.time.load()
    gdp6h_batch = cd.ragged.subset(gdp6h_ds, {"time": (time[0], time[-1])}).drop_vars(["rowsize", "id"])
    gdp6h_batch = interpolate_drifters_location(gdp6h_batch, time, lat_v, lon_u, uv_fields)

    LOGGER.info("2.3. Evaluating SSC against drifters velocities")
    LOGGER.info("2.3.1. Computing along trajectories metrics")
    traj_metrics = compute_along_traj_metrics(gdp6h_batch, uv_fields.keys())
    del gdp6h_batch
    LOGGER.info("2.3.2. Computing binned metrics")
    binned_sums, binned_counts, sum_expression = compute_binned_metrics(duacs_ds, traj_metrics, uv_fields.keys())
    del traj_metrics

    LOGGER.info("2.4. Computing additional kinematics")
    kinematics_vars = compute_kinematics(uv_fields, lat_u, lon_u, lat_v, lon_v, mask)
    methods_batch = list(uv_fields.keys())
    del uv_fields

    # store in datasets
    LOGGER.info("2.4. Storing all in a dataset")
    ssc_fields_ds = _kinematics_to_dataset(kinematics_vars, lat_t, lon_t)
    ssc_fields_ds = _err_to_dataset(ssc_fields_ds, sum_expression, binned_sums, binned_counts)
    ssc_fields_ds = _mask_to_dataset(ssc_fields_ds, mask)
    return ssc_fields_ds, methods_batch
