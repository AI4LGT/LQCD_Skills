"""Minimal resampling companion required by Def_quark_renorm."""

import numpy as np


def jackknife_resampling(data):
    """Return leave-one-configuration-out means along axis zero."""

    if type(data).__module__.split(".")[0] == "cupy":
        import cupy as xp
    else:
        xp = np
    values = xp.asarray(data)
    nconf = values.shape[0]
    if nconf < 2:
        raise ValueError("jackknife_resampling requires at least two configurations")
    return (xp.sum(values, axis=0, keepdims=True) - values) / (nconf - 1)


__all__ = ["jackknife_resampling"]
