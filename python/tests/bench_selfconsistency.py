"""Self-consistency benchmark: gpu_leiden vs. leidenalg seed distribution.

Measures whether the reported gap between gpu_leiden and leidenalg is a real
quality gap or just noise inside leidenalg's own randomized distribution.

For each dataset, runs:
  (A) leidenalg with 10 seeds (s = 0..9)
  (B) gpu_leiden deterministic once (reference)
  (C) gpu_leiden quality with 10 seeds (s = 0..9)

Computes Q distributions, self-consistency ARI matrices, and cross-method ARI.
Saves raw results to /tmp/self_consistency_{name}.pkl.
"""

from __future__ import annotations

import gc
import os
import pickle
import time
from typing import Dict, List, Tuple

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

N_SEEDS = 10
RESOLUTION = 1.0
N_ITERATIONS = 2


def load_connectivity(h5ad_path: str) -> sp.csr_matrix:
    """Load neighbors connectivity (symmetric CSR) from h5ad."""
    adata = sc.read_h5ad(h5ad_path)
    if "connectivities" not in adata.obsp:
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata)
    conn_raw = adata.obsp["connectivities"]
    conn: sp.csr_matrix = sp.csr_matrix(conn_raw)
    conn = (conn + conn.T) / 2.0
    conn.eliminate_zeros()
    del adata
    gc.collect()
    return conn


def csr_to_igraph(csr: sp.csr_matrix) -> ig.Graph:
    """Undirected weighted igraph from symmetric CSR (upper triangle edges)."""
    coo = sp.triu(csr, k=1).tocoo()
    shape = csr.shape
    assert shape is not None
    n = int(shape[0])
    edges = list(zip(coo.row.tolist(), coo.col.tolist()))
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = coo.data.tolist()
    return g


def weighted_q(g: ig.Graph, labels: np.ndarray) -> float:
    return float(g.modularity(labels.tolist(), weights="weight"))


def ncomms(labels: np.ndarray) -> int:
    return int(np.unique(labels).size)


def offdiag_ari(labels_list: List[np.ndarray]) -> Tuple[float, float, np.ndarray]:
    """Compute 10x10 ARI matrix and mean/std of 90 off-diagonal entries."""
    n = len(labels_list)
    mat = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            a = float(adjusted_rand_score(labels_list[i], labels_list[j]))
            mat[i, j] = a
            mat[j, i] = a
    # Off-diagonal entries
    mask = ~np.eye(n, dtype=bool)
    off = mat[mask]
    return float(off.mean()), float(off.std(ddof=0)), mat


def cross_ari(labels_a: List[np.ndarray], labels_b: List[np.ndarray]) -> Tuple[float, float, np.ndarray]:
    """Compute ARI across every pair (a_i, b_j). Returns mean, std, matrix."""
    na, nb = len(labels_a), len(labels_b)
    mat = np.zeros((na, nb), dtype=np.float64)
    for i in range(na):
        for j in range(nb):
            mat[i, j] = float(adjusted_rand_score(labels_a[i], labels_b[j]))
    vals = mat.flatten()
    return float(vals.mean()), float(vals.std(ddof=0)), mat


def run_leidenalg_seed(g: ig.Graph, seed: int) -> Tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        n_iterations=N_ITERATIONS,
        seed=seed,
        resolution_parameter=RESOLUTION,
    )
    dt = time.perf_counter() - t0
    labels = np.asarray(part.membership)
    q = weighted_q(g, labels)
    return labels, q, dt


def run_gpu_det(csr: sp.csr_matrix, g: ig.Graph) -> Tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    labels = gpu_leiden.leiden_from_csr(
        csr,
        flavor="deterministic",
        resolution=RESOLUTION,
        max_iterations=N_ITERATIONS,
        random_seed=42,
        verbose=False,
    )
    dt = time.perf_counter() - t0
    labels = np.asarray(labels)
    q = weighted_q(g, labels)
    return labels, q, dt


def run_gpu_qual_seed(csr: sp.csr_matrix, g: ig.Graph, seed: int) -> Tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    labels = gpu_leiden.leiden_from_csr(
        csr,
        flavor="quality",
        resolution=RESOLUTION,
        max_iterations=N_ITERATIONS,
        random_seed=seed,
        n_restarts=4,
        temperature=0.5,
        verbose=False,
    )
    dt = time.perf_counter() - t0
    labels = np.asarray(labels)
    q = weighted_q(g, labels)
    return labels, q, dt


def summarize_q(qs: List[float]) -> Dict[str, float]:
    arr = np.asarray(qs, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def process_dataset(name: str, path: str) -> Dict:
    print("=" * 72, flush=True)
    print(f"DATASET: {name} -- {path}", flush=True)
    print("=" * 72, flush=True)

    t0 = time.perf_counter()
    csr = load_connectivity(path)
    t_load = time.perf_counter() - t0
    csr_shape = csr.shape
    assert csr_shape is not None
    n_nodes = int(csr_shape[0])
    n_edges_dir = csr.nnz
    print(
        f"[{name}] loaded: nodes={n_nodes}, nnz(directed)={n_edges_dir}, "
        f"load_time={t_load:.2f}s",
        flush=True,
    )

    t0 = time.perf_counter()
    g = csr_to_igraph(csr)
    t_build = time.perf_counter() - t0
    print(
        f"[{name}] igraph built: |V|={g.vcount()}, |E|={g.ecount()}, "
        f"build_time={t_build:.2f}s",
        flush=True,
    )

    # ---- Experiment A: leidenalg N seeds ----
    print(f"[{name}] >>> leidenalg N={N_SEEDS} seeds <<<", flush=True)
    la_labels: List[np.ndarray] = []
    la_qs: List[float] = []
    la_ks: List[int] = []
    la_ts: List[float] = []
    for s in range(N_SEEDS):
        labels, q, dt = run_leidenalg_seed(g, s)
        la_labels.append(labels)
        la_qs.append(q)
        la_ks.append(ncomms(labels))
        la_ts.append(dt)
        print(
            f"[{name}] leidenalg seed={s}: k={la_ks[-1]}, Qw={q:.6f}, time={dt:.2f}s",
            flush=True,
        )

    # ---- Experiment B: gpu_leiden deterministic ----
    print(f"[{name}] >>> gpu_leiden deterministic <<<", flush=True)
    det_labels, det_q, det_t = run_gpu_det(csr, g)
    det_k = ncomms(det_labels)
    print(
        f"[{name}] deterministic: k={det_k}, Qw={det_q:.6f}, time={det_t:.2f}s",
        flush=True,
    )

    # ---- Experiment C: gpu_leiden quality N seeds ----
    print(f"[{name}] >>> gpu_leiden quality N={N_SEEDS} seeds <<<", flush=True)
    qu_labels: List[np.ndarray] = []
    qu_qs: List[float] = []
    qu_ks: List[int] = []
    qu_ts: List[float] = []
    for s in range(N_SEEDS):
        labels, q, dt = run_gpu_qual_seed(csr, g, s)
        qu_labels.append(labels)
        qu_qs.append(q)
        qu_ks.append(ncomms(labels))
        qu_ts.append(dt)
        print(
            f"[{name}] qual seed={s}: k={qu_ks[-1]}, Qw={q:.6f}, time={dt:.2f}s",
            flush=True,
        )

    # ---- Metrics ----
    la_q_summary = summarize_q(la_qs)
    qu_q_summary = summarize_q(qu_qs)

    la_ari_mean, la_ari_std, la_ari_mat = offdiag_ari(la_labels)
    qu_ari_mean, qu_ari_std, qu_ari_mat = offdiag_ari(qu_labels)

    # det vs leidenalg: 10 values
    det_vs_la_vals = np.asarray(
        [float(adjusted_rand_score(det_labels, la_labels[s])) for s in range(N_SEEDS)],
        dtype=np.float64,
    )
    det_vs_la_mean = float(det_vs_la_vals.mean())
    det_vs_la_std = float(det_vs_la_vals.std(ddof=0))

    # qual vs leidenalg: 100 pairs
    qu_vs_la_mean, qu_vs_la_std, qu_vs_la_mat = cross_ari(qu_labels, la_labels)

    # Is det inside leidenalg Q distribution?
    la_q_min = la_q_summary["min"]
    la_q_mean = la_q_summary["mean"]
    la_q_std = la_q_summary["std"]
    if la_q_std > 0:
        det_in_std = (det_q - la_q_mean) / la_q_std
    else:
        det_in_std = float("inf") if det_q != la_q_mean else 0.0
    det_in_dist = (det_q >= la_q_min) and (det_q >= la_q_mean - 2.0 * la_q_std)

    result = {
        "name": name,
        "n_nodes": n_nodes,
        "n_edges_dir": n_edges_dir,
        # leidenalg
        "la_labels": la_labels,
        "la_qs": la_qs,
        "la_ks": la_ks,
        "la_ts": la_ts,
        "la_q_summary": la_q_summary,
        "la_ari_mean": la_ari_mean,
        "la_ari_std": la_ari_std,
        "la_ari_mat": la_ari_mat,
        # det
        "det_labels": det_labels,
        "det_q": det_q,
        "det_k": det_k,
        "det_t": det_t,
        "det_in_dist": det_in_dist,
        "det_in_std": det_in_std,
        # qual
        "qu_labels": qu_labels,
        "qu_qs": qu_qs,
        "qu_ks": qu_ks,
        "qu_ts": qu_ts,
        "qu_q_summary": qu_q_summary,
        "qu_ari_mean": qu_ari_mean,
        "qu_ari_std": qu_ari_std,
        "qu_ari_mat": qu_ari_mat,
        # cross
        "det_vs_la_vals": det_vs_la_vals,
        "det_vs_la_mean": det_vs_la_mean,
        "det_vs_la_std": det_vs_la_std,
        "qu_vs_la_mean": qu_vs_la_mean,
        "qu_vs_la_std": qu_vs_la_std,
        "qu_vs_la_mat": qu_vs_la_mat,
    }

    # Persist raw
    pkl_path = f"/tmp/self_consistency_{name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)
    print(f"[{name}] saved raw results to {pkl_path}", flush=True)

    del csr, g
    gc.collect()
    return result


def print_report(results: Dict[str, Dict]) -> None:
    print()
    print("=" * 72, flush=True)
    print("SELF-CONSISTENCY REPORT", flush=True)
    print("=" * 72, flush=True)

    # Leidenalg Q distribution
    print("\nLeidenalg Q distribution (N=10 seeds):")
    print(f"| {'Dataset':<8} | {'mean':>10} | {'std':>10} | {'min':>10} | {'max':>10} |")
    print("|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|")
    for name, r in results.items():
        s = r["la_q_summary"]
        print(
            f"| {name:<8} | {s['mean']:>10.6f} | {s['std']:>10.6f} | {s['min']:>10.6f} | {s['max']:>10.6f} |"
        )

    # gpu_leiden deterministic single run
    print("\ngpu_leiden determ single run:")
    print(
        f"| {'Dataset':<8} | {'Qw':>10} | {'Ncomms':>6} | {'inside leidenalg Q dist?':<30} |"
    )
    print("|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 8 + "|" + "-" * 32 + "|")
    for name, r in results.items():
        inside = "yes" if r["det_in_dist"] else "no"
        gap = r["det_in_std"]
        note = f"{inside} (mean{gap:+.2f}std)"
        print(
            f"| {name:<8} | {r['det_q']:>10.6f} | {r['det_k']:>6d} | {note:<30} |"
        )

    # gpu_leiden quality distribution
    print("\ngpu_leiden qual Q distribution (N=10 seeds):")
    print(f"| {'Dataset':<8} | {'mean':>10} | {'std':>10} | {'min':>10} | {'max':>10} |")
    print("|" + "-" * 10 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|")
    for name, r in results.items():
        s = r["qu_q_summary"]
        print(
            f"| {name:<8} | {s['mean']:>10.6f} | {s['std']:>10.6f} | {s['min']:>10.6f} | {s['max']:>10.6f} |"
        )

    # Inter-run ARI
    print("\nInter-run ARI:")
    print(f"| {'Comparison':<22} | {'pcw6 ARI':<22} | {'merfish ARI':<22} |")
    print("|" + "-" * 24 + "|" + "-" * 24 + "|" + "-" * 24 + "|")

    def fmt(m: float, s: float) -> str:
        return f"{m:.4f} +/- {s:.4f}"

    def row(label: str, key_mean: str, key_std: str) -> None:
        p = results["pcw6"]
        m = results["merfish"]
        print(
            f"| {label:<22} | {fmt(p[key_mean], p[key_std]):<22} | {fmt(m[key_mean], m[key_std]):<22} |"
        )

    row("leidenalg self-cons", "la_ari_mean", "la_ari_std")
    row("det vs leidenalg", "det_vs_la_mean", "det_vs_la_std")
    row("qual vs leidenalg", "qu_vs_la_mean", "qu_vs_la_std")
    row("qual self-cons", "qu_ari_mean", "qu_ari_std")

    # Concrete per-dataset conclusion
    print()
    print("=" * 72, flush=True)
    print("PER-DATASET VERDICT", flush=True)
    print("=" * 72, flush=True)
    for name, r in results.items():
        print(f"\n[{name}]")
        la_q = r["la_q_summary"]
        qu_q = r["qu_q_summary"]
        det_q = r["det_q"]
        print(
            f"  leidenalg Q:       mean={la_q['mean']:.6f}  std={la_q['std']:.6f}  min={la_q['min']:.6f}  max={la_q['max']:.6f}"
        )
        print(f"  det Q:             {det_q:.6f}  (mean{r['det_in_std']:+.2f} std)")
        print(
            f"  qual Q:            mean={qu_q['mean']:.6f}  std={qu_q['std']:.6f}  min={qu_q['min']:.6f}  max={qu_q['max']:.6f}"
        )
        print(
            f"  leidenalg self-ARI:{r['la_ari_mean']:.4f} +/- {r['la_ari_std']:.4f}"
        )
        print(
            f"  det vs leidenalg:  {r['det_vs_la_mean']:.4f} +/- {r['det_vs_la_std']:.4f}"
        )
        print(
            f"  qual vs leidenalg: {r['qu_vs_la_mean']:.4f} +/- {r['qu_vs_la_std']:.4f}"
        )
        print(
            f"  qual self-ARI:     {r['qu_ari_mean']:.4f} +/- {r['qu_ari_std']:.4f}"
        )


def main() -> None:
    results: Dict[str, Dict] = {}
    for name, path in DATASETS:
        results[name] = process_dataset(name, path)
    print_report(results)


if __name__ == "__main__":
    main()
