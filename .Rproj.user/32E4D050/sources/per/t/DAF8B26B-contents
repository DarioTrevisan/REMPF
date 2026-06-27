from __future__ import annotations
import numpy as np


def summarize_trials(x: np.ndarray, quantiles=(0.1, 0.5, 0.9)) -> dict[str, np.ndarray]:
    """
    Summaries along axis=0 (trials axis). Input x shape (trials, ...).
    Returns mean, std, and quantiles with shape (...) each.
    """
    x = np.asarray(x)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1) if x.shape[0] > 1 else np.zeros_like(mean)

    qs = np.quantile(x, quantiles, axis=0)
    out = {
        "mean": mean,
        "std": std,
        "q_levels": np.array(quantiles, dtype=float),
        "q_values": qs,  # shape (nq_levels, ...)
    }
    return out
