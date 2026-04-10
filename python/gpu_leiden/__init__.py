"""GPU-accelerated Leiden community detection.

This package provides a GPU backend for the Leiden community detection
algorithm, with a minimal API that works directly on scipy CSR sparse
matrices. It's intended as a drop-in accelerator for workflows like
``scanpy.tl.leiden``.

Example
-------
>>> import scipy.sparse
>>> import gpu_leiden
>>> adj = scipy.sparse.csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
>>> labels = gpu_leiden.leiden_from_csr(adj, resolution=1.0)
>>> labels.dtype
dtype('int32')
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._core import leiden_from_csr as _leiden_from_csr_raw  # type: ignore[import-not-found]

__all__ = ["leiden_from_csr", "__version__"]
__version__ = "0.1.0"


def leiden_from_csr(
    adjacency,
    resolution: float = 1.0,
    max_iterations: int = -1,
    random_seed: int = 42,
    flavor: str = "deterministic",
    n_restarts: int = 4,
    temperature: float = 0.5,
) -> np.ndarray:
    """Run GPU Leiden community detection on a sparse graph.

    Parameters
    ----------
    adjacency
        Sparse adjacency matrix (``scipy.sparse`` compatible). Will be
        converted to CSR if it isn't already. For undirected graphs (the
        common case — e.g. scanpy's ``obsp['connectivities']``), the matrix
        should be symmetric.
    resolution
        Resolution (gamma) parameter. Higher values -> more communities.
    max_iterations
        Number of full Leiden passes (each pass = local moving + refinement
        + aggregation hierarchy). Each pass is seeded from the previous
        pass's final partition; the first pass starts from singletons.
        A value of ``-1`` or ``0`` means "use default" (``2``), matching
        leidenalg's default ``n_iterations=2``. Running 2 passes typically
        improves ARI by 0.05-0.15 over a single pass at the cost of roughly
        doubling the runtime.
    random_seed
        Random seed. For ``flavor="deterministic"`` this is reserved for
        future use and does not affect the output (the deterministic path
        is bit-reproducible regardless of seed). For ``flavor="quality"``
        the seed drives the Gumbel-max sampling and iterated local search
        — same seed gives identical labels across runs.
    flavor
        Algorithm flavor:

        * ``"deterministic"`` (default): bit-reproducible GPU Leiden path
          using greedy max-gain phase-1 moves. ~5-12x faster than
          leidenalg but typically reaches 95-99% of leidenalg's
          modularity.
        * ``"quality"``: probabilistic GPU Leiden using Gumbel-max
          sampling in phase-1 and softmax sampling in refinement, plus
          iterated local search over a mandatory deterministic baseline
          plus ``n_restarts`` probabilistic restarts. Each restart runs
          the full hierarchical Leiden; the best unweighted-Q run wins.
          Runtime is typically 3-6x the deterministic flavor (enough to
          stay comparable to leidenalg on larger graphs), and closes
          most of the det-vs-leidenalg modularity / ARI gap on medium
          graphs while guaranteeing no regression versus the
          deterministic baseline.
    n_restarts
        (Quality flavor only.) Number of ILS restarts. Each restart uses
        a different base seed and runs the full hierarchical Leiden; the
        best-modularity restart is returned.
    temperature
        (Quality flavor only.) Gumbel noise scale relative to ``1/weight``
        (the natural unit of Leiden gain). Lower values are greedier
        (closer to deterministic), higher values are more exploratory.
        Default ``0.5`` is empirically a good balance for scanpy-style
        connectivity graphs across a wide range of sizes.

    Returns
    -------
    numpy.ndarray
        ``int32`` array of length ``adjacency.shape[0]`` with one community
        label per node.
    """
    if not sp.issparse(adjacency):
        raise TypeError("adjacency must be a scipy.sparse matrix")

    csr = adjacency.tocsr()
    if csr.shape[0] != csr.shape[1]:
        raise ValueError(
            f"adjacency must be square; got shape {csr.shape}"
        )
    n_nodes = int(csr.shape[0])

    # Map flavor string to the int code expected by the C API.
    flavor_l = str(flavor).lower()
    if flavor_l in ("deterministic", "det", "greedy", "d"):
        flavor_int = 0
    elif flavor_l in ("quality", "prob", "probabilistic", "q"):
        flavor_int = 1
    else:
        raise ValueError(
            f"flavor must be 'deterministic' or 'quality'; got {flavor!r}"
        )

    # Ensure contiguous + correct dtypes. Since the C API takes int (int32)
    # and double (float64) pointers, we force those dtypes here.
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)

    return _leiden_from_csr_raw(
        indptr=indptr,
        indices=indices,
        data=data,
        n_nodes=n_nodes,
        resolution=float(resolution),
        max_iterations=int(max_iterations),
        random_seed=int(random_seed),
        flavor=int(flavor_int),
        n_restarts=int(n_restarts),
        temperature=float(temperature),
    )
