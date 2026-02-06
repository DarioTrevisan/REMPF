from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

def plot_cost_hist(costs: np.ndarray, *, title: str, out: str) -> None:
    plt.figure()
    plt.hist(costs, bins=30)
    plt.title(title)
    plt.xlabel("cost")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
