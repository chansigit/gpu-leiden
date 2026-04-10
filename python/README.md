# gpu-leiden

GPU-accelerated Leiden community detection for large sparse graphs.

## Install

Requires CUDA toolkit (nvcc) and a CUDA-capable GPU.

    pip install -e .

## Usage

    import scipy.sparse
    import gpu_leiden

    adj = scipy.sparse.random(10000, 10000, density=0.001, format='csr')
    adj = (adj + adj.T) / 2   # make symmetric
    labels = gpu_leiden.leiden_from_csr(adj, resolution=1.0)

Returns an `int32` numpy array of community labels.

## Status

Alpha. Single entry point (`leiden_from_csr`). Built as the GPU backend
for single-cell analysis pipelines (e.g. scanpy).
