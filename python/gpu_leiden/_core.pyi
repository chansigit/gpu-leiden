"""Type stubs for the compiled _core extension module."""

from numpy import float64, int32
from numpy.typing import NDArray

def leiden_from_csr(
    indptr: NDArray[int32],
    indices: NDArray[int32],
    data: NDArray[float64],
    n_nodes: int,
    resolution: float = ...,
    max_iterations: int = ...,
    random_seed: int = ...,
    flavor: int = ...,
    n_restarts: int = ...,
    temperature: float = ...,
    verbose: int = ...,
) -> NDArray[int32]: ...
