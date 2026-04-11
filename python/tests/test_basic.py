"""Basic sanity tests for gpu_leiden."""

import numpy as np
import scipy.sparse as sp
import pytest

import gpu_leiden


def _two_cliques(n_per=20):
    """Build a 2-clique symmetric graph with a thin bridge. Should give 2 communities."""
    n = 2 * n_per
    row, col, data = [], [], []
    # Two dense cliques
    for c in range(2):
        offset = c * n_per
        for i in range(n_per):
            for j in range(i + 1, n_per):
                row.append(offset + i); col.append(offset + j); data.append(1.0)
                row.append(offset + j); col.append(offset + i); data.append(1.0)
    # A single bridge edge between the two cliques
    row.append(0); col.append(n_per); data.append(0.1)
    row.append(n_per); col.append(0); data.append(0.1)
    return sp.csr_matrix((data, (row, col)), shape=(n, n))


def test_returns_int32_array():
    n_per = 10
    adj = _two_cliques(n_per=n_per)
    labels = gpu_leiden.leiden_from_csr(adj, resolution=1.0)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int32
    assert labels.shape == (2 * n_per,)


def test_recovers_two_cliques():
    """With two well-separated cliques, should find 2 communities."""
    adj = _two_cliques(n_per=20)
    labels = gpu_leiden.leiden_from_csr(adj, resolution=1.0)
    unique = np.unique(labels)
    assert len(unique) == 2, f"Expected 2 communities, got {len(unique)}: {unique}"
    # Each clique should be one community
    assert len(np.unique(labels[:20])) == 1
    assert len(np.unique(labels[20:])) == 1
    # And the two communities should be different
    assert labels[0] != labels[20]


def test_resolution_changes_granularity():
    """Higher resolution should give more (or equal) communities than lower."""
    adj = _two_cliques(n_per=15)
    labels_low = gpu_leiden.leiden_from_csr(adj, resolution=0.5)
    labels_high = gpu_leiden.leiden_from_csr(adj, resolution=2.0)
    n_low = len(np.unique(labels_low))
    n_high = len(np.unique(labels_high))
    assert n_high >= n_low, f"high res ({n_high}) should be >= low res ({n_low})"


def test_rejects_non_sparse():
    with pytest.raises(TypeError):
        gpu_leiden.leiden_from_csr(np.eye(5))


def test_rejects_non_square():
    with pytest.raises(ValueError):
        gpu_leiden.leiden_from_csr(sp.csr_matrix((3, 4), dtype=float))
