from hydra_zen import store, make_custom_builds_fn

from .data_structure import DataStructure
from .s3filesystem import S3FileSystem


__all__ = ["data_structure_store"]


builds = make_custom_builds_fn(populate_full_signature=True)

s3_fs_conf = builds(S3FileSystem)

local_experiment_conf = builds(DataStructure)
s3_experiment_conf = builds(DataStructure, filesystem=s3_fs_conf)

data_structure_store = store(group="data_structure")
data_structure_store(local_experiment_conf, name="local")
data_structure_store(s3_experiment_conf, name="s3")
