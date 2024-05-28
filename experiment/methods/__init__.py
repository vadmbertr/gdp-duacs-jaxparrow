from hydra_zen import store, make_custom_builds_fn

from . import cyclogeostrophy, geostrophy


__all__ = ["method_store"]


pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

var_conf = pbuilds(cyclogeostrophy.cyclogeostrophy, method="variational")
it_conf = pbuilds(cyclogeostrophy.cyclogeostrophy, method="iterative")
it_filter_conf = pbuilds(cyclogeostrophy.cyclogeostrophy, method="iterative", it_use_filter=True)
geos_conf = pbuilds(geostrophy.geostrophy)

method_store = store(group="method")
method_store(var_conf, name="variational")
method_store(it_conf, name="iterative")
method_store(it_filter_conf, name="iterative_filter")
method_store(geos_conf, name="geostrophy")
