from __future__ import annotations

from typing import Mapping

import pandas as pd


def metrics_dict_to_dataframe(
    metrics: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Convert nested mapping model->metric->value into a DataFrame.

    Rows correspond to models, columns to metric names.
    """
    frame = pd.DataFrame.from_dict(
        {model: dict(mdict) for model, mdict in metrics.items()},
        orient="index",
    )
    frame.index.name = "model"
    return frame


def compare_models_table(
    baseline: str,
    metrics: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    """Return a table of metric deltas relative to the baseline model."""
    df = metrics_dict_to_dataframe(metrics)
    if baseline not in df.index:
        raise KeyError(f"Baseline model {baseline!r} not found in metrics.")
    base_row = df.loc[baseline]
    deltas = df.subtract(base_row, axis="columns")
    # Attach a multi-index (value, delta) per metric
    cols = []
    for metric in df.columns:
        cols.extend(
            [
                (metric, "value"),
                (metric, "delta_vs_baseline"),
            ]
        )
    multi = pd.MultiIndex.from_tuples(cols, names=["metric", "type"])
    out = pd.DataFrame(index=df.index, columns=multi, dtype=float)
    for metric in df.columns:
        out[(metric, "value")] = df[metric]
        out[(metric, "delta_vs_baseline")] = deltas[metric]
    return out
