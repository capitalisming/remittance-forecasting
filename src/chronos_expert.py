"""E3: Chronos-Bolt zero-shot expert. No fitting step — pure inference on the same real-time
information set (`realtime.info_set(v)`) the ridge experts see, so it is directly comparable and
just as leak-safe. See docs/limitations note on TSFM pretraining-corpus contamination."""
import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

import panel
import realtime

_PIPE = None
MODEL_NAME = "amazon/chronos-bolt-small"
# Pinned to the exact HF Hub commit this study's results were produced with (obtained 2026-09-24
# via HfApi().model_info("amazon/chronos-bolt-small").sha), so a future push to the model repo
# can't silently change results out from under the paper's numbers.
MODEL_REVISION = "772f3d25d38aec6d914c8949dab4462e2d46f5d8"


def get_pipeline():
    global _PIPE
    if _PIPE is None:
        _PIPE = BaseChronosPipeline.from_pretrained(MODEL_NAME, revision=MODEL_REVISION,
                                                      device_map="cpu", torch_dtype=torch.float32)
    return _PIPE


def chronos_forecast(v, min_history=13):
    hist = realtime.info_set(v)["value"]
    target_month = v.to_period("M") + 1
    if len(hist) < min_history:
        return None, target_month
    ctx = torch.tensor(hist.to_numpy(), dtype=torch.float32)
    quantiles, _ = get_pipeline().predict_quantiles(inputs=ctx, prediction_length=1, quantile_levels=[0.5])
    return float(quantiles[0, 0, 0].item()), target_month


def backtest_chronos(issue_dates):
    truth = panel.first_release_series()
    rows = []
    for v in issue_dates:
        pred, target_month = chronos_forecast(v)
        if pred is None:
            continue
        rows.append(dict(issue_date=v, target_month=target_month, forecast=pred,
                         actual=truth.get(target_month, np.nan)))
    out = pd.DataFrame(rows).set_index("issue_date")
    out["abs_err"] = (out.forecast - out.actual).abs()
    return out
