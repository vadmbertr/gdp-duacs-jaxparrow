import jaxparrow as jpw
from jaxtyping import Array, Float

from .map import parallelize_and_vectorize


def geostrophy(
        lat_t: Float[Array, "lat lon"], lon_t: Float[Array, "lat lon"],
        adt_t: Float[Array, "time lat lon"], mask: Float[Array, "time lat lon"]
) -> [Float[Array, "(time) lat lon"], ...]:
    # convenient wrapper to avoid too much verbosity when using vmap and shmap
    def partial_geostrophy(partial_adt_t, partial_mask):
        return jpw.geostrophy(partial_adt_t, lat_t, lon_t, partial_mask)
    # parallelize and vectorize over the time dimension
    pv_geos_fn = parallelize_and_vectorize(partial_geostrophy)

    return pv_geos_fn([adt_t, mask])
