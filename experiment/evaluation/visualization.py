import os

from cartopy import crs as ccrs
import cmocean.cm as cmo
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..filesystem.data_structure import DataStructure
from ..logger.logger import LOGGER


SPEED_CMAP = cmo.speed  # noqa
# SPEED_CMAP.set_bad(color="lightgrey")
CURL_CMAP = cmo.curl  # noqa
# CURL_CMAP.set_bad(color="lightgrey")
MATTER_CMAP = cmo.matter  # noqa
# MATTER_CMAP.set_bad(color="lightgrey")
AMP_CMAP = cmo.amp  # noqa
# AMP_CMAP.set_bad(color="lightgrey")
BALANCE_R_CMAP = cmo.balance_r  # noqa
# BALANCE_R_CMAP.set_bad(color="lightgrey")
HALINE_CMAP = cmo.haline  # noqa
# HALINE_CMAP.set_bad(color="lightgrey")


def _plot(
        ssc_fields_ds: xr.Dataset, data_var: str,
        cmap: mpl.colors.LinearSegmentedColormap, vmax: float, vmin: float = None, cmap_centered: bool = False
) -> plt.Figure:
    if vmax is None:
        vmax = _get_max(ssc_fields_ds, [data_var], apply_abs=cmap_centered)
    if cmap_centered:
        vmin = -vmax
    elif vmin is None:
        vmin = 0

    field = ssc_fields_ds[data_var]

    fig, ax = plt.subplots(figsize=(13, 4), subplot_kw={"projection": ccrs.PlateCarree()})

    ax.set_title(field.attrs["method"])
    im = ax.pcolormesh(ssc_fields_ds.longitude, ssc_fields_ds.latitude, field,
                       cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(),
                       linewidth=0, rasterized=True)
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    clb = fig.colorbar(im, cax=cax, label=field.attrs["what"])
    clb.ax.set_title(field.attrs["units"])
    ax.coastlines()

    return fig


def _save_fig(
        ssc_fields_ds: xr.Dataset, data_var: str,
        data_structure: DataStructure, plot_path: str,
        cmap: mpl.colors.LinearSegmentedColormap, vmax: float = None, vmin: float = None, cmap_centered: bool = False
):
    try:
        fig = _plot(ssc_fields_ds, data_var, cmap, vmax, vmin, cmap_centered)
        filepath = os.path.join(plot_path, f"{data_var}.pdf")
        try:
            f = data_structure.open(filepath, "wb")
            fig.savefig(f, format="pdf")
            f.close()
        except Exception as e:
            LOGGER.warning(e)
            fig.savefig(filepath, format="pdf")
        plt.close()
    except Exception as e:
        LOGGER.warning(e)


def _get_max(ssc_fields_ds: xr.Dataset, data_vars: [], apply_abs: bool = False) -> float:
    try:
        if apply_abs:
            ds = abs(ssc_fields_ds[data_vars])
        else:
            ds = ssc_fields_ds[data_vars]
        vmax = float(ds.max().to_dataarray().max())
    except Exception as e:
        LOGGER.warning(e)
        vmax = None
    return vmax


def _get_quantile(ssc_fields_ds: xr.Dataset, data_vars: [], quantile: float, apply_abs: bool = False) -> float:
    try:
        if apply_abs:
            ds = abs(ssc_fields_ds[data_vars])
        else:
            ds = ssc_fields_ds[data_vars]
        data = ds.to_array().values.ravel()
        vmax = np.quantile(data[np.isfinite(data)], quantile)
    except Exception as e:
        LOGGER.warning(e)
        vmax = None
    return vmax


def _get_vars(ssc_fields_ds: xr.Dataset, contains: str, excludes: str = None) -> []:
    data_vars = ssc_fields_ds.data_vars
    return [
        data_var for data_var in data_vars
        if (contains in data_var) and ((excludes is None) or (excludes not in data_var))
    ]


def _plot_kinematics(ssc_fields_ds: xr.Dataset, data_structure: DataStructure, plot_path: str, quantile: float = None):
    u_vars = _get_vars(ssc_fields_ds, contains="u_avg_", excludes="_diff_")
    v_vars = _get_vars(ssc_fields_ds, contains="v_avg_", excludes="_diff_")
    magn_vars = _get_vars(ssc_fields_ds, contains="magn_avg_", excludes="_diff_")
    nrv_vars = _get_vars(ssc_fields_ds, contains="nrv_avg_", excludes="_diff_")
    eke_vars = _get_vars(ssc_fields_ds, contains="eke_avg_", excludes="_diff_")

    if quantile is None:
        vmax_fn = _get_max
    else:
        vmax_fn = lambda f, v, apply_abs=False: _get_quantile(f, v, quantile, apply_abs)

    u_max = vmax_fn(ssc_fields_ds, u_vars)
    v_max = vmax_fn(ssc_fields_ds, v_vars)
    magn_max = vmax_fn(ssc_fields_ds, magn_vars)
    nrv_max = vmax_fn(ssc_fields_ds, nrv_vars, apply_abs=True)
    eke_max = vmax_fn(ssc_fields_ds, eke_vars)

    for data_var in u_vars:
        _save_fig(ssc_fields_ds, data_var, data_structure, plot_path, SPEED_CMAP, vmax=u_max)
    for data_var in v_vars:
        _save_fig(ssc_fields_ds, data_var, data_structure, plot_path, SPEED_CMAP, vmax=v_max)
    for data_var in magn_vars:
        _save_fig(ssc_fields_ds, data_var, data_structure, plot_path, SPEED_CMAP, vmax=magn_max)
    for data_var in nrv_vars:
        _save_fig(ssc_fields_ds, data_var, data_structure, plot_path, CURL_CMAP, vmax=nrv_max, cmap_centered=True)
    for data_var in eke_vars:
        _save_fig(ssc_fields_ds, data_var, data_structure, plot_path, MATTER_CMAP, vmax=eke_max)


def _plot_errors(ssc_fields_ds: xr.Dataset, data_structure: DataStructure, plot_path: str, quantile: float = None):
    err_u_avg_vars = _get_vars(ssc_fields_ds, contains="err_u_avg_", excludes="_diff_")
    err_v_avg_vars = _get_vars(ssc_fields_ds, contains="err_v_avg_", excludes="_diff_")
    err_avg_vars = _get_vars(ssc_fields_ds, contains="err_avg_", excludes="_diff_")

    if quantile is None:
        vmax_fn = _get_max
    else:
        vmax_fn = lambda f, v=False: _get_quantile(f, v, quantile)

    err_u_avg_max = vmax_fn(ssc_fields_ds, err_avg_vars)
    err_v_avg_max = vmax_fn(ssc_fields_ds, err_avg_vars)
    err_avg_max = vmax_fn(ssc_fields_ds, err_avg_vars)

    for data_var in err_u_avg_vars:
        _save_fig(ssc_fields_ds, data_var, data_structure, plot_path, AMP_CMAP, vmax=err_u_avg_max)
    for data_var in err_v_avg_vars:
        _save_fig(ssc_fields_ds, data_var, data_structure, plot_path, AMP_CMAP, vmax=err_v_avg_max)
    for data_var in err_avg_vars:
        _save_fig(ssc_fields_ds, data_var, data_structure, plot_path, AMP_CMAP, vmax=err_avg_max)


def _plot_differences(ssc_fields_ds: xr.Dataset, data_structure: DataStructure, plot_path: str,
                      quantile: float = None):
    def do_plot(contains: str, vmax: float = None):
        if "_rel_" in contains:
            excludes = None
        else:
            excludes = "_rel_"
        eke_vars = _get_vars(ssc_fields_ds, contains=f"eke_avg{contains}", excludes=excludes)
        err_u_avg_vars = _get_vars(ssc_fields_ds, contains=f"err_u_avg{contains}", excludes=excludes)
        err_v_avg_vars = _get_vars(ssc_fields_ds, contains=f"err_v_avg{contains}", excludes=excludes)
        err_avg_vars = _get_vars(ssc_fields_ds, contains=f"err_avg{contains}", excludes=excludes)

        for data_var in eke_vars + err_u_avg_vars + err_v_avg_vars + err_avg_vars:
            if quantile is not None:
                vmax = _get_quantile(ssc_fields_ds, [data_var], quantile, apply_abs=True)
            _save_fig(ssc_fields_ds, data_var, data_structure, plot_path, BALANCE_R_CMAP, vmax=vmax, cmap_centered=True)

    do_plot(contains="_diff_")
    do_plot(contains="_diff_rel_")


def _plot_ssh(ssh_fields_ds: xr.Dataset, data_structure: DataStructure, plot_path: str):
    data_var = "adt"

    vmax = np.nanmax(ssh_fields_ds[data_var])
    vmin = np.nanmin(ssh_fields_ds[data_var])

    _save_fig(ssh_fields_ds, data_var, data_structure, plot_path, HALINE_CMAP, vmax=vmax, vmin=vmin)


def plot_fields(ssc_fields_ds: xr.Dataset, data_structure: DataStructure, experiment_path: str):
    plot_path = os.path.join(experiment_path, "plots")
    data_structure.makedirs(plot_path)

    _plot_kinematics(ssc_fields_ds, data_structure, plot_path, quantile=.999)
    _plot_errors(ssc_fields_ds, data_structure, plot_path, quantile=.999)
    _plot_differences(ssc_fields_ds, data_structure, plot_path, quantile=.999)
    # _plot_ssh(ssc_fields_ds, data_structure, plot_path)
