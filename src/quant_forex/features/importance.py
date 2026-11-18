"""Univariate association of features at t with the next-bar return.

This is not model feature importance, not causal proof, and not a reason to trade.
Permutation importance belongs with fitted models in a later phase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def next_bar_associations(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    if "close" not in frame.columns:
        raise ValueError("close is required to form the next-bar return.")
    forward = frame["close"].pct_change().shift(-1)
    rows: list[dict[str, object]] = []
    for column in feature_columns:
        paired = pd.DataFrame({"x": frame[column], "y": forward}).dropna()
        n = int(len(paired))
        if n < 8 or paired["x"].nunique() < 2:
            rows.append(
                {
                    "feature": column,
                    "n": n,
                    "pearson_corr_with_next_return": None,
                    "abs_corr": None,
                }
            )
            continue
        corr = float(np.corrcoef(paired["x"].to_numpy(), paired["y"].to_numpy())[0, 1])
        rows.append(
            {
                "feature": column,
                "n": n,
                "pearson_corr_with_next_return": corr,
                "abs_corr": abs(corr),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("abs_corr", ascending=False, na_position="last").reset_index(drop=True)
