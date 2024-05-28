import logging
from collections.abc import Callable
import os

from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.conf import HydraConf, JobConf
from hydra.experimental.callback import Callback
from hydra_zen import store, zen, make_custom_builds_fn

from experiment.batching.process import estimate_and_evaluate
from experiment.evaluation.visualization import plot_fields
from experiment.filesystem import data_structure_store  # noqa
from experiment.filesystem.data_structure import DataStructure
from experiment.filesystem.zarr_store import ZarrStore
from experiment.inputs import duacs_conf, gdp6h_store  # noqa
from experiment.inputs.duacs import DuacsDs
from experiment.inputs.gdp6h import GDP6hDs
from experiment.logger.logger import LOGGER
from experiment.methods import method_store  # noqa
from experiment.preproc import preproc_store  # noqa


class EnvVar(Callback):
    def __init__(self, *, env_file_path: str = ".env"):
        if not os.path.exists(env_file_path):
            env_file_path = ""
        self.env_file_path = env_file_path

    def on_job_start(self, **kw):
        load_dotenv(self.env_file_path, override=True)  # load env. var. (for credentials) from file if provided


@store(
    name="gdp6h_duacs_jaxparrow",
    hydra_defaults=[
        "_self_",
        {"data_structure": "s3"},
        {"gdp6h_ds": "global"},
        {"gdp6h_preproc": "base"}
    ],
    logger_level=logging.DEBUG,
    duacs_ds=duacs_conf,
    methods={
        "Variational": method_store["method", "variational"],
        # "Iterative": method_store["method", "iterative"],
        # "Iterative_filter": method_store["method", "iterative_filter"],
        "Geostrophy": method_store["method", "geostrophy"]
    }
)
def experiment(
        logger_level: int,
        data_structure: DataStructure,
        gdp6h_ds: GDP6hDs,
        gdp6h_preproc: Callable,
        duacs_ds: DuacsDs,
        methods: dict,
        memory_per_device: int = 35  # in Gb
):
    LOGGER.setLevel(logger_level)
    # 0. init data structure
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    experiment_path = os.path.join(
        data_structure.root_path,
        os.path.basename(os.path.dirname(hydra_output_dir)),
        os.path.basename(hydra_output_dir)
    )  # reconstruct hydra experiment path structure
    # 0.1. experiment data (zarr input & output)
    gdp6h_zstore = ZarrStore(data_structure.data_path, data_structure.gdp6h_zarr_dir, data_structure.filesystem)
    scc_fields_zstore = ZarrStore(experiment_path, "ssc_fields.zarr", data_structure.filesystem)
    # 0.2. copy experiment config
    try:
        data_structure.copy(os.path.join(hydra_output_dir, ".hydra"), experiment_path)
    except FileExistsError:
        LOGGER.debug(f"{hydra_output_dir} already exists")

    LOGGER.info("1. Loading input datasets")
    LOGGER.info("1.1.1. Loading GDP6H")
    gdp6h_ds.load_data(gdp6h_zstore)
    LOGGER.info("1.1.2. Preprocessing GDP6H")
    LOGGER.debug(
        f"before preprocessing: "
        f"{int(gdp6h_ds.dataset.traj.size)} drifters & "
        f"{int(gdp6h_ds.dataset.obs.size)} observations"
    )
    gdp6h_ds.apply_preproc(gdp6h_preproc)
    LOGGER.debug(
        f"after preprocessing: "
        f"{int(gdp6h_ds.dataset.traj.size)} drifters & "
        f"{int(gdp6h_ds.dataset.obs.size)} observations"
    )
    LOGGER.info("1.2. Loading DUACS")
    duacs_ds.load_data(gdp6h_ds)
    LOGGER.debug(
        f"DUACS domain dimensions: {duacs_ds.dataset.dims}"
    )

    LOGGER.info("2. Estimating and evaluating SSC methods in mini-batch")
    ssc_fields_ds = estimate_and_evaluate(gdp6h_ds, duacs_ds, methods, memory_per_device)

    LOGGER.info("3. Producing plots")
    plot_fields(ssc_fields_ds, data_structure, experiment_path)

    LOGGER.info("4. Saving time averaged SSC fields dataset")
    ssc_fields_ds.to_zarr(scc_fields_zstore.store, mode="w")


if __name__ == "__main__":
    builds = make_custom_builds_fn(populate_full_signature=True)
    store(HydraConf(job=JobConf(chdir=False), callbacks={"env_var": builds(EnvVar)}))

    store.add_to_hydra_store()

    zen(experiment).hydra_main(
        config_name="gdp6h_duacs_jaxparrow",
        version_base="1.1",
        config_path=".",
    )
