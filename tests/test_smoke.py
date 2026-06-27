import numpy as np
from rempf.core.matching.exact import bipartite_matching_cost_exact

def test_exact_cost_nonnegative():
    rng = np.random.default_rng(0)
    x = rng.random((20, 2))
    y = rng.random((20, 2))
    c = bipartite_matching_cost_exact(x, y, p=1.0)
    print(c)
    assert c >= 0.0


def test_torus_distance_wraps() -> None:
    x = np.array([[0.95, 0.50]])
    y = np.array([[0.05, 0.50]])

    cube_cost = bipartite_matching_cost_exact(x, y, p=1.0, domain="cube")
    torus_cost = bipartite_matching_cost_exact(x, y, p=1.0, domain="torus")

    assert np.isclose(cube_cost, 0.9)
    assert np.isclose(torus_cost, 0.1)
