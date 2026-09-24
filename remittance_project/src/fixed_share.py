"""Recursive fixed-share aggregation over the four candidates (naive, expanding ridge,
rolling ridge w=48, Chronos zero-shot).

Split in two phases so hyperparameter search doesn't refit models:
  1) precompute_forecasts(): each expert's h=1 forecast at every issue date (the expensive part).
  2) simulate(): pure arithmetic recursive weight update over the precomputed archive (cheap),
     so eta/alpha can be swept freely.

Feedback timing: month m's outcome is used to update weights only once m appears in
realtime.info_set(v) for the current issue date v -- the same real-time delay the forecasts
themselves respect (Experiment A / k=0 "operational" condition). Experiment D (extra k-month
delay) is a thin wrapper added later.
"""
import numpy as np
import pandas as pd

import chronos_expert as ce
import experts
import panel
import realtime

EXPERTS = ["naive", "expanding_ridge", "rolling_ridge_w48", "chronos"]
K = len(EXPERTS)
MIN_TRAIN_ROWS = 13


def first_valid_issue_date(issue_dates, min_train=MIN_TRAIN_ROWS):
    for v in issue_dates:
        try:
            _, y, _, _ = panel.build_row(v)
        except AssertionError:
            continue
        if len(y) >= min_train:
            return v
    raise RuntimeError("no issue date reaches the minimum training size")


def expert_forecast(name, v):
    if name == "chronos":
        return ce.chronos_forecast(v, min_history=MIN_TRAIN_ROWS)
    X, y, q, target_month = panel.build_row(v)
    if len(y) < MIN_TRAIN_ROWS:
        return None, target_month
    if name == "naive":
        pred = experts.seasonal_naive(X, y, q, target_month=target_month)
    elif name == "expanding_ridge":
        pred = experts.expanding_ridge(X, y, q)
    elif name == "rolling_ridge_w48":
        pred = experts.rolling_ridge(X, y, q, window=48)
    else:
        raise ValueError(name)
    return pred, target_month


def precompute_forecasts(issue_dates):
    """target_month -> {expert_name: point_forecast}"""
    archive = {}
    for v in issue_dates:
        row = {}
        for k in EXPERTS:
            pred, target_month = expert_forecast(k, v)
            row[k] = pred
        archive[target_month] = row
    return archive


def simulate(archive, issue_dates, eta, alpha, clip_c, delay_k=0):
    """delay_k: extra months of artificial delay added on top of the real publication lag,
    for Experiment D. delay_k=0 reproduces the natural operational timing (Experiment A)."""
    truth = panel.first_release_series()
    weights = {k: 1.0 / K for k in EXPERTS}
    known_months, log = set(), []

    for v in issue_dates:
        target_month = v.to_period("M") + 1
        info_v = v - pd.DateOffset(months=delay_k) if delay_k else v
        info = realtime.info_set(info_v)
        newly_eligible = sorted(m for m in archive if m not in known_months and m in info.index)

        for m in newly_eligible:
            y_true = truth.get(m)
            known_months.add(m)
            if y_true is None or pd.isna(y_true):
                continue
            preds_m = archive[m]
            losses = np.array([
                1.0 if (preds_m.get(k) is None or pd.isna(preds_m.get(k)))
                else min(abs(preds_m[k] - y_true) / clip_c, 1.0)
                for k in EXPERTS
            ])
            w = np.array([weights[k] for k in EXPERTS])
            q = w * np.exp(-eta * losses)
            q = q / q.sum()
            p_new = (1 - alpha) * q + alpha / K
            weights = dict(zip(EXPERTS, p_new))

        preds = archive.get(target_month, {})
        valid = {k: p for k, p in preds.items() if p is not None and not pd.isna(p)}
        wsum = sum(weights[k] for k in valid)
        combined = sum(weights[k] * valid[k] for k in valid) / wsum if valid and wsum > 0 else np.nan
        log.append(dict(issue_date=v, target_month=target_month, combined_forecast=combined,
                        actual=truth.get(target_month, np.nan),
                        **{f"w_{k}": weights[k] for k in EXPERTS}))

    out = pd.DataFrame(log).set_index("issue_date")
    out["abs_err"] = (out.combined_forecast - out.actual).abs()
    return out


def equal_weight_baseline(archive, issue_dates):
    truth = panel.first_release_series()
    rows = []
    for v in issue_dates:
        target_month = v.to_period("M") + 1
        preds = archive.get(target_month, {})
        valid = [p for p in preds.values() if p is not None and not pd.isna(p)]
        combined = float(np.mean(valid)) if valid else np.nan
        rows.append(dict(issue_date=v, target_month=target_month, combined_forecast=combined,
                         actual=truth.get(target_month, np.nan)))
    out = pd.DataFrame(rows).set_index("issue_date")
    out["abs_err"] = (out.combined_forecast - out.actual).abs()
    return out
