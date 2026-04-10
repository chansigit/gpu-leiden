#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstdint>
#include <stdexcept>

namespace nb = nanobind;

// Forward declaration of the C API (defined in leiden.cu)
extern "C" int leiden_from_csr(
    const int*    out_indptr,
    const int*    out_indices,
    const double* out_data,
    int n_out_edges,
    const int*    in_indptr,
    const int*    in_indices,
    const double* in_data,
    int n_in_edges,
    int n_nodes,
    double resolution,
    int max_iterations,
    unsigned int random_seed,
    int* out_labels
);

// Python wrapper accepting numpy arrays for a symmetric (undirected) CSR.
// For scanpy's use case the connectivities matrix is symmetric, so the same
// CSR is used for both in- and out-adjacency.
nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>
leiden_from_csr_py(
    nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig> indptr,
    nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig> indices,
    nb::ndarray<const double,  nb::ndim<1>, nb::c_contig> data,
    int n_nodes,
    double resolution,
    int max_iterations,
    unsigned int random_seed)
{
    if (indptr.shape(0) != (size_t)(n_nodes + 1)) {
        throw std::invalid_argument("indptr length must be n_nodes + 1");
    }
    if (indices.shape(0) != data.shape(0)) {
        throw std::invalid_argument("indices and data must have the same length");
    }

    int n_edges = (int)indices.shape(0);

    // Allocate output labels; ownership transferred to numpy via capsule.
    int* labels = new int[n_nodes];

    // Call C API. Use the same CSR for in- and out-adjacency (symmetric case).
    int rc = leiden_from_csr(
        indptr.data(), indices.data(), data.data(), n_edges,
        indptr.data(), indices.data(), data.data(), n_edges,
        n_nodes,
        resolution,
        max_iterations,
        random_seed,
        labels);

    if (rc != 0) {
        delete[] labels;
        throw std::runtime_error("leiden_from_csr returned non-zero status");
    }

    size_t shape[1] = { (size_t)n_nodes };
    nb::capsule owner(labels, [](void* p) noexcept { delete[] (int*)p; });
    return nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(
        labels, 1, shape, owner);
}

NB_MODULE(_core, m) {
    m.doc() = "GPU Leiden — low-level bindings";
    m.def("leiden_from_csr",
          &leiden_from_csr_py,
          nb::arg("indptr"),
          nb::arg("indices"),
          nb::arg("data"),
          nb::arg("n_nodes"),
          nb::arg("resolution") = 1.0,
          nb::arg("max_iterations") = -1,
          nb::arg("random_seed") = 0,
          "Run GPU Leiden on a symmetric CSR sparse graph.\n"
          "Returns an int32 numpy array of length n_nodes with community labels.");
}
