import os
import shutil

from .s3filesystem import S3FileSystem


class DataStructure:
    def __init__(
            self,
            data_path: str = "data",
            gdp6h_zarr_dir: str = "data.zarr",
            root_path: str = "experiments",
            filesystem: S3FileSystem = None
    ):
        self.data_path = data_path
        self.gdp6h_zarr_dir = gdp6h_zarr_dir
        self.root_path = os.path.join(root_path, "gdp6h_duacs_jaxparrow")
        self.filesystem = filesystem

    def copy(self, lpath, rpath):
        rpath = f"{rpath}/"
        if self.filesystem is None:
            shutil.copytree(lpath, rpath)
        else:
            self.filesystem.put(lpath, rpath, recursive=True)

    def makedirs(self, path):
        if self.filesystem is None:
            os.makedirs(path, exist_ok=True)
        else:
            self.filesystem.makedirs(path, exist_ok=True)

    def open(self, path: str, mode: str):
        if self.filesystem is None:
            f = open(path, mode=mode)
        else:
            f = self.filesystem.open(path, mode=mode)
        return f
