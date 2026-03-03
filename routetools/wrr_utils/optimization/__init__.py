"""Minimal optimization package exports for benchmark usage.

Only expose Circumnavigate (and Node) to avoid importing heavy/unused
optimizers when `from routetools.wrr_utils.optimization import Circumnavigate`
is executed by the benchmark entrypoint.
"""

from routetools.wrr_utils.optimization.astar.circumnavigate import Circumnavigate, Node

__all__ = ["Circumnavigate", "Node"]
