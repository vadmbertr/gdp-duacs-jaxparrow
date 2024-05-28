from hydra_zen import store, make_custom_builds_fn

from .duacs import DuacsDs
from .gdp6h import GDP6hDs, mediterranean_masking, alboran_masking


__all__ = ["duacs_conf", "gdp6h_store"]


def mediterranean_wrapper(GDP6hDsCls):
    def wrapper(*args, **kwargs):
        ds = GDP6hDsCls(*args, **kwargs)
        ds.filter_fn = mediterranean_masking
        ds.name = "mediterranea"
        return ds

    return wrapper


def alboran_wrapper(GDP6hDsCls):
    def wrapper(*args, **kwargs):
        ds = GDP6hDsCls(*args, **kwargs)
        ds.filter_fn = alboran_masking
        ds.name = "alboran"
        return ds

    return wrapper


builds = make_custom_builds_fn(populate_full_signature=True)

gdp6h_conf = builds(GDP6hDs)
gdp6h_med_conf = builds(GDP6hDs, zen_wrappers=mediterranean_wrapper)
gdp6h_alboran_conf = builds(GDP6hDs, zen_wrappers=alboran_wrapper)

gdp6h_store = store(group="gdp6h_ds")
gdp6h_store(gdp6h_conf, name="global")
gdp6h_store(gdp6h_med_conf, name="mediterranea")
gdp6h_store(gdp6h_alboran_conf, name="alboran")

duacs_conf = builds(DuacsDs)
