# gpu-leiden

GPU-accelerated Leiden community detection for large sparse graphs.

Python bindings (via `nanobind` + `scikit-build-core`) around the
CUDA C++ core at [`chansigit/gpu-leiden`](https://github.com/chansigit/gpu-leiden).
Designed as a drop-in backend for `scanpy.tl.leiden`: takes a
`scipy.sparse` CSR matrix (typically `adata.obsp['connectivities']`)
and returns community labels.

## Install

Requires CUDA toolkit (nvcc), a CUDA-capable GPU (sm_80+), and
Python >= 3.10.

```bash
pip install -e .
```

No PyPI wheels yet — the editable install builds against the local
CUDA toolkit and compute capability.

## Quick usage

```python
import scipy.sparse as sp
import gpu_leiden

adj = sp.csr_matrix(...)  # symmetric connectivity matrix
labels = gpu_leiden.leiden_from_csr(
    adj,
    resolution=1.0,
    random_seed=42,
    flavor="deterministic",   # or "quality"
)
```

## Flavors

| Flavor | Guarantee |
|---|---|
| `deterministic` (default) | Bit-reproducible. 95–99% of leidenalg's modularity, 3x–12x faster. |
| `quality` | Shake-kick ILS; never worse than deterministic. Closes most of the gap to leidenalg on scanpy graphs. Reproducible given `random_seed`. |

See the main repository README for the full API, benchmark numbers,
and build-from-source instructions.

## API

```python
gpu_leiden.leiden_from_csr(
    adjacency,                # scipy.sparse, will be coerced to CSR
    resolution=1.0,
    max_iterations=-1,        # -1 -> default 2 (matches leidenalg)
    random_seed=42,
    flavor="deterministic",   # "deterministic" | "quality"
    n_restarts=4,             # quality flavor only
    temperature=0.5,          # quality flavor only
    verbose=False,            # re-enable developer-facing C output
) -> np.ndarray  # int32, shape (n_nodes,)
```

## Status

Alpha. Suitable for single-cell clustering pipelines; API is stable
between v0.3-quality and any future v0.3.x patch releases but may
evolve across minor versions.
