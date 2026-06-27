from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_edge_dist(*, edge_lengths: np.ndarray, out: str, bins: int, ccdf: bool, title: str) -> None:
    x = np.asarray(edge_lengths, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]

    plt.figure()
    if ccdf:
        xs = np.sort(x)
        n = xs.size
        # CCDF: P(L >= t)
        cc = 1.0 - (np.arange(n) / n)
        plt.plot(xs, cc)
        plt.yscale("log")
        plt.xlabel(r"scaled edge length  $n^{1/d}\|x-y\|$")
        plt.ylabel(r"CCDF  $\mathbb{P}(n^{1/d}\|x-y\| > t)$")
    else:
        plt.hist(x, bins=bins, density=True)
        plt.xlabel(r"scaled edge length  $n^{1/d}\|x-y\|$")
        plt.ylabel("density")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
