import numpy as np
from rempf.core.matching.exact import bipartite_matching_cost_exact

def test_exact_cost_nonnegative():
    rng = np.random.default_rng(0)
    x = rng.random((20, 2))
    y = rng.random((20, 2))
    c = bipartite_matching_cost_exact(x, y, p=1.0)
    assert c >= 0.0
