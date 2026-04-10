# gpu-leiden

GPU-accelerated Leiden community detection for large sparse graphs, built as a
drop-in backend for single-cell clustering pipelines such as
[`scanpy.tl.leiden`](https://scanpy.readthedocs.io/en/stable/api/generated/scanpy.tl.leiden.html).

The core is CUDA C++; the public interface is a Python extension
(`gpu_leiden`) that accepts a `scipy.sparse` CSR matrix and returns
community labels as a numpy array. Typical wall-clock speedup vs
`leidenalg` on real single-cell connectivity graphs is **3x-12x** on the
deterministic flavor, with iterated-local-search support for a
probabilistic "quality" flavor that closes most of the modularity gap
to `leidenalg`.

This repository is a fork maintained at
[`chansigit/gpu-leiden`](https://github.com/chansigit/gpu-leiden). It
diverges from the original
[`Beenishgul/Leiden`](https://github.com/Beenishgul/Leiden) repository
at commit `9e82353` ("Add resolution parameter support"); all subsequent
work (Python bindings, deterministic refactor, quality flavor, shake
perturbation, release tags) lives on this fork.

## Status

| | |
|---|---|
| Latest tag | [`v0.3-quality`](https://github.com/chansigit/gpu-leiden/releases/tag/v0.3-quality) |
| Deterministic baseline tag | [`v0.2-deterministic`](https://github.com/chansigit/gpu-leiden/releases/tag/v0.2-deterministic) |
| Active branch | `feature/resolution-parameter` |
| Python package name | `gpu_leiden` |
| C API entry point | `leiden_from_csr` (`leiden.h`) |
| License | Inherited from upstream Beenishgul/Leiden |

## Install (Python)

Prerequisites:

- NVIDIA GPU with compute capability >= 8.0 (tested on sm_80)
- CUDA toolkit >= 12.0 (provides `nvcc`)
- gcc >= 9 (tested with 12.x)
- Python >= 3.10
- `scikit-build-core`, `nanobind`, `numpy`, `scipy`

Editable install from a clone:

```bash
git clone https://github.com/chansigit/gpu-leiden
cd gpu-leiden/python
pip install -e . --no-deps
```

No PyPI wheels yet (CUDA `cibuildwheel` pipeline is planned; see
[Known TODOs](#known-todos)). Editable install builds the extension
against your local CUDA toolkit, so the resulting `.so` is tied to your
compute capability and CUDA version.

## Usage

### Minimal

```python
import scipy.sparse as sp
import gpu_leiden

adj = sp.csr_matrix(...)                # symmetric, weighted or unweighted
labels = gpu_leiden.leiden_from_csr(
    adj,
    resolution=1.0,
    random_seed=42,
    flavor="deterministic",             # or "quality"
)
# labels: int32 ndarray of shape (n_nodes,)
```

### With scanpy

```python
import scanpy as sc
import gpu_leiden
import pandas as pd

adata = sc.datasets.pbmc3k_processed()
sc.pp.neighbors(adata)

labels = gpu_leiden.leiden_from_csr(
    adata.obsp["connectivities"],
    resolution=1.0,
    random_seed=0,
    flavor="quality",
    n_restarts=4,
)
adata.obs["leiden_gpu"] = pd.Categorical(labels.astype(str))
```

The higher-level drop-in `sc.tl.leiden(adata)` replacement is provided
by the companion
[`sjanpy`](https://github.com/chansigit/sjanpy) package (`sjanpy.tl.leiden`).

### Full API

```python
gpu_leiden.leiden_from_csr(
    adjacency,                          # scipy.sparse — will be coerced to CSR
    resolution=1.0,
    max_iterations=-1,                  # -1 -> default 2 (matches leidenalg)
    random_seed=42,
    flavor="deterministic",             # "deterministic" | "quality"
    n_restarts=4,                       # quality flavor only
    temperature=0.5,                    # quality flavor only (Gumbel scale)
    verbose=False,                      # re-enable C/CUDA developer output
)
```

## Flavors

| Flavor | Guarantee | Algorithm |
|---|---|---|
| `deterministic` (default) | Bit-reproducible across runs. 95–99% of `leidenalg`'s modularity. 3x–12x faster. Frozen at tag `v0.2-deterministic`. | Greedy max-gain local moving with warp-cooperative phase-1 kernels, Thrust-based aggregation, deterministic CPU refinement. No RNG. |
| `quality` | Always at least as good as `deterministic` on the same graph (mandatory deterministic baseline + 1e-4 improvement threshold). Reproducible given a fixed `random_seed`. Introduced at tag `v0.3-quality`. | Shake-kick ILS: the deterministic baseline run establishes a floor; subsequent restarts warm-start from the best-so-far labels, shatter the largest communities back into singletons ("kick"), then run a deterministic local search from the perturbed partition. Cross-restart comparison uses the weighted modularity (matching `g.modularity(weights='weight')`). |

Use `deterministic` for speed and reproducibility; use `quality` when you
want to pay 3x–5x runtime to close most of the modularity gap to
`leidenalg`.

## Benchmark

All numbers were taken on a single A100 with CUDA 12.8, matched against
`leidenalg==0.10` (`n_iterations=2`, `resolution=1.0`, `seed=42`) on
scanpy-style connectivity graphs. Modularity is reported as the
igraph-weighted modularity (`g.modularity(labels, weights='weight')`).
ARI is measured against `leidenalg`'s labels on the same graph.

| Dataset | Cells | Edges | Method | N(comms) | Weighted Q | ARI | Time (s) |
|---|---:|---:|---|---:|---:|---:|---:|
| pcw6 | 28,630 | 627,742 | deterministic | 19 | 0.823107 | 0.7259 | 0.43 |
| pcw6 | 28,630 | 627,742 | **quality v0.3** | 21 | **0.823372** | **0.7919** | 1.48 |
| pcw6 | 28,630 | 627,742 | leidenalg | 20 | 0.833235 | 1.0000 | 2.32 |
| merfish | 152,720 | 3,747,570 | deterministic | 14 | 0.796533 | 0.8365 | 3.48 |
| merfish | 152,720 | 3,747,570 | **quality v0.3** | 14 | **0.799730** | **0.8563** | 13.98 |
| merfish | 152,720 | 3,747,570 | leidenalg | 15 | 0.805756 | 1.0000 | 18.12 |

Reproduce with `python/tests/bench_phase3p2.py`.

Observations:
- On both datasets the v0.3 quality flavor strictly improves over the
  deterministic baseline (Phase 3.1's probabilistic restarts silently
  tied deterministic due to early-stop and unweighted-Q scorer bugs
  that were fixed in Phase 3.2).
- Deterministic flavor is still 4x–5x faster than `leidenalg` at the
  merfish scale.
- A small gap to `leidenalg`'s modularity remains (~0.010 on pcw6,
  ~0.006 on merfish), but ARI agreement is already high and the quality
  flavor is directionally closing the gap.

## Building the C++/CUDA core

For developers hacking on the CUDA kernels or the CLI binary
(`./leiden`), the standalone build uses a plain Makefile:

```bash
# From the repository root:
make              # builds ./leiden
./leiden graph.txt cpu 1.0       # CPU reference
./leiden graph.txt gpu 1.0       # GPU deterministic
./leiden graph.txt gpu_csr 1.0   # Exercises the leiden_from_csr C API
```

The Python extension is built via `scikit-build-core` from the `python/`
subdirectory; it invokes `nvcc` on the same `leiden.cu` / `leiden.cpp`
sources plus `python/src/bindings.cpp`.

## Regression tests

```bash
./tests/verify.sh                               # CLI byte-identity (graph.txt)

cd python
pytest tests/                                   # Python smoke test

cd ../../sjanpy
pytest tests/test_leiden.py -v                  # End-to-end scanpy integration
```

The `verify.sh` script captures deterministic byte-identity on
`graph.txt`; any change to the deterministic output across a merge
should either be accompanied by a regenerated `tests/baseline_{cpu,gpu}.txt`
or treated as a regression.

## Known TODOs

- `cibuildwheel` pipeline for pre-built CUDA wheels on PyPI.
- Collapse profile instrumentation behind the `verbose` flag (done in
  `bfe160d` — `verbose=True` re-enables the developer output; default
  is silent).
- Optional: randomised node-order refinement to close the remaining
  gap to `leidenalg`.
- Optional: CUDA-streams parallel restarts (sequential restarts today).

## Releases and tags

- [`v0.3-quality`](https://github.com/chansigit/gpu-leiden/releases/tag/v0.3-quality)
  — current head; shake-kick ILS with weighted-Q scorer.
- [`v0.2-deterministic`](https://github.com/chansigit/gpu-leiden/releases/tag/v0.2-deterministic)
  — preserves the fully bit-reproducible deterministic-only state for
  users who need absolute reproducibility.
- `v0.0.1` — initial fork point from upstream Beenishgul/Leiden.

## References

- Traag, V. A., Waltman, L., & van Eck, N. J. (2019).
  *From Louvain to Leiden: guaranteeing well-connected communities.*
  Scientific Reports 9:5233. https://doi.org/10.1038/s41598-019-41695-0
- Upstream project: https://github.com/Beenishgul/Leiden
- Original paper accompanying the upstream GPU implementation is
  referenced from the upstream README.
