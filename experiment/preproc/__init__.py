from hydra_zen import store, make_custom_builds_fn
import toolz
import xarray as xr

from . import gdp6h


__all__ = ["preproc_store"]


GDP6H_STEPS = (
    "gps_only",
    "after_2000_only",
    "svp_only",
    "before_2023_06_07_only",
    "drogued_only",
    "remove_low_latitudes",
    "finite_value_only",
    "remove_outlier_values"
)


def apply_preproc(ds: xr.Dataset, steps: tuple):
    return toolz.compose_left(*[getattr(gdp6h, step) for step in steps])(ds)


pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

gdp6h_preproc_conf = pbuilds(apply_preproc, steps=GDP6H_STEPS)

preproc_store = store(group="gdp6h_preproc")
preproc_store(gdp6h_preproc_conf, name="base")
