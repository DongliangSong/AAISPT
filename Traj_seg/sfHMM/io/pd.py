from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import sfHMMBase
from ..single_sfhmm import sfHMM1

__all__ = ["save"]


def save(obj: sfHMMBase, path: str) -> None:
    """
    Save obj.step.fit, obj.data_fil and obj.viterbi as csv.

    Parameters
    ----------
    obj : sfHMMBase
        sfHMM object to save.
    path : str
        Saving path.
    """
    df_list = _to_dataframes(obj)
    out = pd.concat(df_list, axis=1)
    out.to_csv(path)
    return None


def _to_dataframes(obj: sfHMMBase, suffix: str = "") -> list[pd.DataFrame]:
    if isinstance(obj, sfHMM1):
        df = pd.DataFrame(data=obj.data_raw, dtype=np.float64,
                          columns=[f"data_raw{suffix}"],
                          index=np.arange(obj.data_raw.size, dtype=np.int32))
        if obj.step is not None:
            df[f"step finding{suffix}"] = obj.step.fit
        if obj.data_fil is not None:
            df[f"denoised{suffix}"] = obj.data_fil
        if obj.viterbi is not None:
            df[f"Viterbi path{suffix}"] = obj.viterbi
        return [df]
    else:
        raise TypeError(f"Only sfHMM objects can be converted to pd.DataFrame, but got {type(obj)}")
