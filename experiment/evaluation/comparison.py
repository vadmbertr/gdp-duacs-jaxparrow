import xarray as xr


def _differences(ref: xr.DataArray, other: xr.DataArray) -> (xr.DataArray, xr.DataArray):
    abs_diff = other - ref
    rel_diff = 100 * abs_diff / other  # .where(other > 1e-3, 0)
    return abs_diff, rel_diff


def _compare(ssc_fields_ds: xr.Dataset, data_vars: dict, ref_method: str, method: str) -> dict:
    abs_diff, rel_diff = _differences(ssc_fields_ds[f"eke_avg_{ref_method}"], ssc_fields_ds[f"eke_avg_{method}"])
    data_vars[f"eke_avg_diff_{ref_method}_{method}"] = -abs_diff  # note the "-"
    data_vars[f"eke_avg_diff_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$m^2/s^{-2}$",
        "what": "$\\langle \\text{EKE}_1 \\rangle - \\langle \\text{EKE}_2 \\rangle$"
    }
    data_vars[f"eke_avg_diff_rel_{ref_method}_{method}"] = -rel_diff  # note the "-"
    data_vars[f"eke_avg_diff_rel_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$\\%$",
        "what": "$100 (\\langle \\text{EKE}_1 \\rangle - \\langle \\text{EKE}_2 \\rangle) / "
                "\\langle \\text{EKE}_2 \\rangle$"
    }

    abs_diff, rel_diff = _differences(ssc_fields_ds[f"err_u_avg_{ref_method}"], ssc_fields_ds[f"err_u_avg_{method}"])
    data_vars[f"err_u_avg_diff_{ref_method}_{method}"] = abs_diff
    data_vars[f"err_u_avg_diff_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$m/s$",
        "what": "$\\langle \\epsilon_{2_u} \\rangle - \\langle \\epsilon_{1_u} \\rangle$"
    }
    data_vars[f"err_u_avg_diff_rel_{ref_method}_{method}"] = rel_diff
    data_vars[f"err_u_avg_diff_rel_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$\\%$",
        "what": "$100 (\\langle \\epsilon_{2_u} \\rangle - \\langle \\epsilon_{1_u} \\rangle) / "
                "\\langle \\epsilon_{2_u} \\rangle$"
    }

    abs_diff, rel_diff = _differences(ssc_fields_ds[f"err_v_avg_{ref_method}"], ssc_fields_ds[f"err_v_avg_{method}"])
    data_vars[f"err_v_avg_diff_{ref_method}_{method}"] = abs_diff
    data_vars[f"err_v_avg_diff_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$m/s$",
        "what": "$\\langle \\epsilon_{2_v} \\rangle - \\langle \\epsilon_{1_v} \\rangle$"
    }
    data_vars[f"err_v_avg_diff_rel_{ref_method}_{method}"] = rel_diff
    data_vars[f"err_v_avg_diff_rel_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$\\%$",
        "what": "$100 (\\langle \\epsilon_{2_v} \\rangle - \\langle \\epsilon_{1_v} \\rangle) / "
                "\\langle \\epsilon_{1_v} \\rangle$"
    }

    abs_diff, rel_diff = _differences(ssc_fields_ds[f"err_avg_{ref_method}"], ssc_fields_ds[f"err_avg_{method}"])
    data_vars[f"err_avg_diff_{ref_method}_{method}"] = abs_diff
    data_vars[f"err_avg_diff_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$m/s$",
        "what": "$\\langle \\epsilon_2 \\rangle - \\langle \\epsilon_1 \\rangle$"
    }
    data_vars[f"err_avg_diff_rel_{ref_method}_{method}"] = rel_diff
    data_vars[f"err_avg_diff_rel_{ref_method}_{method}"].attrs = {
        "method": f"{ref_method}(1) ; {method}(2)", "units": "$\\%$",
        "what": "$100 (\\langle \\epsilon_2 \\rangle - \\langle \\epsilon_1 \\rangle) / \\langle \\epsilon_2 \\rangle$"
    }

    return data_vars


def compare_methods(ssc_fields_ds: xr.Dataset, methods: []) -> xr.Dataset:
    if "Variational" in methods:
        ref_method = "Variational"
        other_methods = ["DUACS"] + list(set(methods) - {ref_method})
    elif "Geostrophy" in methods:
        ref_method = "Geostrophy"
        other_methods = ["DUACS"] + list(set(methods) - {ref_method})
    else:
        ref_method = "DUACS"
        other_methods = methods

    data_vars = {}
    for method in other_methods:
        data_vars = _compare(ssc_fields_ds, data_vars, ref_method, method)

    if "Geostrophy" in methods and ref_method == "Variational":
        data_vars = _compare(ssc_fields_ds, data_vars, "Geostrophy", "DUACS")

    return ssc_fields_ds.assign(**data_vars)
