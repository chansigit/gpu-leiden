"""Phase 3.2 benchmark: ILS + shake perturbation on pcw6 and merfish.

For each dataset, runs:
  (a) gpu_leiden deterministic
  (b) gpu_leiden quality (Phase 3.2, n_restarts=4, temperature=0.5)
  (c) leidenalg reference
and reports #communities, weighted Q, wall-time, and ARI vs leidenalg.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from typing import Tuple

import numpy as np
import scipy.sparse as sp

os.environ.setdefault("SCANPY_VERBOSITY", "0")

import scanpy as sc  # noqa: E402
import igraph as ig  # noqa: E402
import leidenalg  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

import gpu_leiden  # noqa: E402


DATASETS = [
    (
        "pcw6",
        "/scratch/users/chensj16/transfer/engreitz-lab/VIC-analysis-V250428/adata.pcw6.h5ad",
    ),
    (
        "merfish",
        "/scratch/users/chensj16/transfer/engreitz-lab/VIC-analysis-V250428/ventricle_13pcw_merfish.h5ad",
    ),
]


def load_connectivity(h5ad_path: str) -> sp.csr_matrix:
    """Load the neighbors connectivity matrix from an h5ad file.

    Uses pre-computed obsp['connectivities'] if available; otherwise runs
    sc.pp.neighbors with default parameters.
    """
    adata = sc.read_h5ad(h5ad_path)
    if "connectivities" not in adata.obsp:
        # Ensure there's something to compute on
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata)
    conn_raw = adata.obsp["connectivities"]
    conn: sp.csr_matrix = sp.csr_matrix(conn_raw)
    # Symmetrize in case it's not already
    conn = (conn + conn.T) / 2.0
    conn.eliminate_zeros()
    del adata
    gc.collect()
    return conn


def csr_to_igraph(csr: sp.csr_matrix) -> ig.Graph:
    """Wrap a symmetric CSR as an undirected weighted igraph.Graph.

    Uses the upper triangle so each undirected edge appears once.
    """
    coo = sp.triu(csr, k=1).tocoo()
    shape = csr.shape
    assert shape is not None
    n = int(shape[0])
    edges = list(zip(coo.row.tolist(), coo.col.tolist()))
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = coo.data.tolist()
    return g


def run_gpu(
    csr: sp.csr_matrix,
    flavor: str,
    **kwargs,
) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    labels = gpu_leiden.leiden_from_csr(
        csr,
        resolution=1.0,
        max_iterations=2,
        random_seed=42,
        flavor=flavor,
        **kwargs,
    )
    t1 = time.perf_counter()
    return np.asarray(labels), t1 - t0


def run_leidenalg(g: ig.Graph) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        n_iterations=2,
        seed=42,
        resolution_parameter=1.0,
    )
    t1 = time.perf_counter()
    return np.asarray(part.membership), t1 - t0


def weighted_q(g: ig.Graph, labels: np.ndarray) -> float:
    return float(g.modularity(labels.tolist(), weights="weight"))


def ncomms(labels: np.ndarray) -> int:
    return int(np.unique(labels).size)


def main() -> None:
    rows = []
    for name, path in DATASETS:
        print("=" * 72, flush=True)
        print(f"DATASET: {name} -- {path}", flush=True)
        print("=" * 72, flush=True)

        t0 = time.perf_counter()
        csr = load_connectivity(path)
        t_load = time.perf_counter() - t0
        csr_shape = csr.shape
        assert csr_shape is not None
        n_nodes = int(csr_shape[0])
        n_edges_dir = csr.nnz  # directed-count
        print(
            f"[{name}] loaded: nodes={n_nodes}, nnz(directed)={n_edges_dir}, "
            f"load_time={t_load:.2f}s",
            flush=True,
        )

        # Build igraph for leidenalg + weighted Q
        t0 = time.perf_counter()
        g = csr_to_igraph(csr)
        t_build = time.perf_counter() - t0
        print(
            f"[{name}] igraph built: |V|={g.vcount()}, |E|={g.ecount()}, "
            f"build_time={t_build:.2f}s",
            flush=True,
        )

        # --- leidenalg reference ---
        print(f"[{name}] >>> leidenalg reference <<<", flush=True)
        lref_labels, lref_t = run_leidenalg(g)
        lref_q = weighted_q(g, lref_labels)
        lref_k = ncomms(lref_labels)
        print(
            f"[{name}] leidenalg: k={lref_k}, Qw={lref_q:.6f}, time={lref_t:.2f}s",
            flush=True,
        )

        # --- gpu_leiden deterministic ---
        print(f"[{name}] >>> gpu_leiden deterministic <<<", flush=True)
        det_labels, det_t = run_gpu(csr, flavor="deterministic")
        det_q = weighted_q(g, det_labels)
        det_k = ncomms(det_labels)
        det_ari = float(adjusted_rand_score(lref_labels, det_labels))
        print(
            f"[{name}] deterministic: k={det_k}, Qw={det_q:.6f}, "
            f"time={det_t:.2f}s, ARI={det_ari:.4f}",
            flush=True,
        )

        # --- gpu_leiden quality (Phase 3.2) ---
        print(f"[{name}] >>> gpu_leiden quality Phase 3.2 <<<", flush=True)
        sys.stdout.flush()
        qual_labels, qual_t = run_gpu(
            csr,
            flavor="quality",
            n_restarts=4,
            temperature=0.5,
        )
        sys.stdout.flush()
        qual_q = weighted_q(g, qual_labels)
        qual_k = ncomms(qual_labels)
        qual_ari = float(adjusted_rand_score(lref_labels, qual_labels))
        print(
            f"[{name}] quality: k={qual_k}, Qw={qual_q:.6f}, "
            f"time={qual_t:.2f}s, ARI={qual_ari:.4f}",
            flush=True,
        )

        rows.append(
            (name, "deterministic", det_k, det_q, det_t, det_ari)
        )
        rows.append(
            (name, "quality (3.2)", qual_k, qual_q, qual_t, qual_ari)
        )
        rows.append(
            (name, "leidenalg", lref_k, lref_q, lref_t, 1.0)
        )

        del csr, g
        gc.collect()

    print()
    print("=" * 72, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 72, flush=True)
    header = f"| {'Dataset':<8} | {'Method':<14} | {'Ncomms':>6} | {'Weighted Q':>10} | {'Time (s)':>8} | {'ARI vs leidenalg':>16} |"
    sep = "|" + "-" * (len(header) - 2) + "|"
    print(header)
    print(sep)
    for name, method, k, q, tsec, ari in rows:
        print(
            f"| {name:<8} | {method:<14} | {k:>6d} | {q:>10.6f} | {tsec:>8.2f} | {ari:>16.4f} |"
        )


if __name__ == "__main__":
    main()
