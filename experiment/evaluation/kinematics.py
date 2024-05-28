import jax
from jaxparrow.tools.kinematics import magnitude, normalized_relative_vorticity
from jaxtyping import Array, Float
import numpy as np

from ..evaluation.interpolation import interpolate_grid


def compute_kinematics(
        uv_fields: dict,
        lat_u: Float[Array, "lat lon"],
        lon_u: Float[Array, "lat lon"],
        lat_v: Float[Array, "lat lon"],
        lon_v: Float[Array, "lat lon"],
        mask: Float[Array, "time lat lon"]
) -> dict:
    def nrv(u, v, m):
        return normalized_relative_vorticity(u, v, lat_u, lat_v, lon_u, lon_v, m)

    vmap_magn = jax.vmap(magnitude, in_axes=(0, 0))
    vmap_nrv = jax.vmap(nrv, in_axes=(0, 0, 0))

    kinematics_vars = {}
    for method, uv in uv_fields.items():
        kinematics_vars[f"magn_{method}"] = np.array(vmap_magn(*uv), dtype=np.float32)
        kinematics_vars[f"nrv_{method}"] = np.array(vmap_nrv(*uv, mask), dtype=np.float32)
        kinematics_vars[f"u_{method}"] = np.array(interpolate_grid(uv[0], axis=1, padding="left"), dtype=np.float32)
        kinematics_vars[f"v_{method}"] = np.array(interpolate_grid(uv[1], axis=0, padding="left"), dtype=np.float32)
        kinematics_vars[f"u_sq_{method}"] = kinematics_vars[f"u_{method}"] ** 2
        kinematics_vars[f"v_sq_{method}"] = kinematics_vars[f"v_{method}"] ** 2

    return kinematics_vars
