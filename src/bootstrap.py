"""Paired moving-block bootstrap over origins, block lengths {3,6,12} reported as a sensitivity
(not a single choice), per the evaluation plan. Pairing is preserved: each bootstrap draw resamples
whole blocks of (issue_date -> paired difference) so the same origins are used for both methods
being compared in that draw.
"""
import numpy as np


def moving_block_bootstrap_mean(diffs: np.ndarray, block_length: int, n_boot=5000, seed=0):
    """diffs: 1D array of per-origin paired differences, in chronological order.
    Returns the bootstrap distribution of the mean (length n_boot)."""
    n = len(diffs)
    block_length = min(block_length, n)
    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start + 1, size=(n_boot, n_blocks))
    means = np.empty(n_boot)
    offsets = np.arange(block_length)
    for i in range(n_boot):
        idx = (starts[i][:, None] + offsets[None, :]).ravel()[:n]
        means[i] = diffs[idx].mean()
    return means


def summarize(diffs: np.ndarray, label: str, block_lengths=(3, 6, 12), n_boot=5000):
    point = diffs.mean()
    print(f"{label}: point estimate (mean paired diff) = {point:+.2f}, n_origins={len(diffs)}")
    for bl in block_lengths:
        boot = moving_block_bootstrap_mean(diffs, bl, n_boot=n_boot)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        excludes_zero = (lo > 0) or (hi < 0)
        print(f"  block={bl:>2}mo: 95% CI [{lo:+.2f}, {hi:+.2f}]"
              f"{'  <- excludes 0' if excludes_zero else '  (includes 0, not distinguishable from no difference)'}")
    return point
