"""Oracle error aggregation (deterministic simplified variant)."""

from __future__ import annotations

import numpy as np


def compute_oracle_error(render_err: np.ndarray, geom_err: np.ndarray, topo_err: np.ndarray) -> np.ndarray:
    """Combine render/geometry/topology errors with fixed weights.

    This is a placeholder for v0.1 and intentionally simple.
    """
    render_err = np.asarray(render_err, dtype=float).reshape(-1)
    geom_err = np.asarray(geom_err, dtype=float).reshape(-1)
    topo_err = np.asarray(topo_err, dtype=float).reshape(-1)
    n = min(render_err.size, geom_err.size, topo_err.size)
    if n == 0:
        return np.zeros((0,), dtype=float)

    combined = 0.6 * render_err[:n] + 0.3 * geom_err[:n] + 0.1 * topo_err[:n]
    combined = np.clip(combined, 0.0, None)
    return combined
