from datetime import timedelta
from typing import Literal

from jax import vmap
from jaxparrow.tools import operators as jpw_operators
from jaxtyping import Array, Float
import numpy as np
from parcels import FieldSet, JITParticle, ParticleSet
from parcels.tools.statuscodes import StatusCode
import xarray as xr


def interpolate_grid(
        field: Float[Array, "lat lon"],
        axis: Literal[0, 1],
        padding: Literal["left", "right"]
) -> np.ndarray:
    interpolation_map = vmap(jpw_operators.interpolation, in_axes=(0, None, None))
    return np.array(interpolation_map(field, axis, padding), dtype=np.float32)


def interpolate_drifters_location(
        gdp6h_ds: xr.Dataset,
        ssc_time: Float[Array, "time"],
        latitude_v: Float[Array, "lat lon"],
        longitude_u: Float[Array, "lat lon"],
        uv_fields_hat: dict
) -> xr.Dataset:
    def interp_fn(particle, fieldset, time):
        u, v = fieldset.UV[time, 0, particle.lat, particle.lon]  # !!! returns deg/s (m/s not possible in JIT mode)
        particle.U = u
        particle.V = v
        particle.state = StatusCode.StopExecution  # only sample once

    # to m/s
    u_converter_fn = lambda val, lat: val * 1000. * 1.852 * 60. * np.cos(lat * np.pi / 180)
    v_converter_fn = lambda val: val * 1000. * 1.852 * 60.

    dimensions = {
        "time": ssc_time,
        "lat": latitude_v,
        "lon": longitude_u  # parcels needs lon/lat at the f (corner) points of the C-grid
    }
    interp_method = {
        "U": "cgrid_velocity",
        "V": "cgrid_velocity"
    }
    # particle class
    SampleParticle = JITParticle
    SampleParticle = SampleParticle.add_variable("U", dtype=np.float32, initial=np.nan)
    SampleParticle = SampleParticle.add_variable("V", dtype=np.float32, initial=np.nan)

    for method, (u_field, v_field) in uv_fields_hat.items():
        data = {"U": u_field, "V": v_field}
        field = FieldSet.from_data(data=data, dimensions=dimensions, interp_method=interp_method)

        pset = ParticleSet.from_list(field, SampleParticle, lon=gdp6h_ds.lon, lat=gdp6h_ds.lat, time=gdp6h_ds.time)

        pset.execute(interp_fn, dt=timedelta(hours=6), endtime=gdp6h_ds.time[-1].values + np.timedelta64(6, "h"))

        gdp6h_ds[f"u_hat_{method}"] = ("obs", u_converter_fn(pset.U, gdp6h_ds.lat.data))
        gdp6h_ds[f"v_hat_{method}"] = ("obs", v_converter_fn(pset.V))

    return gdp6h_ds.drop_vars(["time"])
