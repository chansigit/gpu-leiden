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

from ._core import leiden_from_csr as _leiden_from_csr_raw

__all__ = ["leiden_from_csr", "__version__"]
__version__ = "0.1.0"


def leiden_from_csr(
    adjacency,
    resolution: float = 1.0,
    max_iterations: int = -1,
    random_seed: int = 0,
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
        Resolution (gamma) parameter. Higher values → more communities.
    max_iterations
        Number of full Leiden passes (each pass = local moving + refinement
        + aggregation hierarchy). Each pass is seeded from the previous
        pass's final partition; the first pass starts from singletons.
        A value of ``-1`` or ``0`` means "use default" (``2``), matching
        leidenalg's default ``n_iterations=2``. Running 2 passes typically
        improves ARI by 0.05-0.15 over a single pass at the cost of roughly
        doubling the runtime.
    random_seed
        Reserved for a future randomized initialization. Currently ignored.

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
    )
