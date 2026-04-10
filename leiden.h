// leiden.h

#ifndef LEIDEN_H
#define LEIDEN_H
#include <stdio.h>
#include "struct.h"

struct gpu_partition {
  double* tot_out;
  double* tot_in;
  double* sum_in;
  double* in_deg;
  double* out_deg;
  int* node_comm;
  int* final_comm;
  int* size;
  int* home_comm;
  int* older_comm;
  double* sum_kin;
  double* self_loops;
  double weight;
  size_t node_comm_size;
  int *nbrs;
  int *pos;
   int *neigh_commNb;
    int *neigh_pos;
};
  struct gpu_graph {
  int* child_in;
  int* child_out;
  double* wts_out;
  double* wts_in;
  int* in_col;
  int* out_col;
  int nodes;
};


Leiden_Partition create_partition(graph& g, Leiden_Partition& p);
int Leiden_GPU(Leiden_Partition& p, graph& g, int E,
               int* tracked_labels, int n_original,
               int flavor, unsigned int random_seed,
               double temperature, double* out_modularity);
double find_quality_cpu(Leiden_Partition& p, graph& g);
void refine_partition_cpu(Leiden_Partition& p, graph& g);
// Probabilistic variant: samples from softmax(gain/temperature) over
// positive-gain candidates via Gumbel-max trick. seed determines the
// per-parent-community RNG streams for reproducibility.
void refine_partition_cpu_prob(Leiden_Partition& p, graph& g,
                               unsigned int seed, double temperature);
// Like create_c_partition, but initializes node_comm[i] = initial_labels[i]
// instead of singleton i. Used by the n_iterations outer loop in
// leiden_from_csr so that subsequent iterations start from the previous
// iteration's final partition, matching leidenalg's default behaviour.
Leiden_Partition create_c_partition_from_labels(graph& g, Leiden_Partition& p,
                                                const int* initial_labels);
double ToOwnCommunity(int node, int community, double bestGain, int old_comm, Leiden_Partition& d_p, graph& d_g);
double computGain(int node, int community, Leiden_Partition& d_p, graph& d_g);
double find_to_own(Leiden_Partition& d_p, graph& d_g, double dncomm, int i, int community, int comm);

extern "C" {

int leiden_from_csr(
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
    // Number of full Leiden passes (local moving + refinement + aggregation
    // hierarchy). Each pass is seeded from the previous pass's final
    // partition. A value <= 0 means "use default" (2, matching leidenalg).
    int max_iterations,
    unsigned int random_seed,
    int* out_labels,
    // Phase 3.1: probabilistic "quality" flavor.
    //   flavor = 0 -> deterministic path (bit-reproducible, v0.2 behaviour)
    //   flavor = 1 -> quality path (Gumbel-max sampled moves + iterated
    //                 local-search over n_restarts seeds; returns the
    //                 best-modularity labels across restarts)
    int flavor,
    // Number of restarts used by the quality flavor. Ignored for the
    // deterministic flavor. A value <= 0 means "use default" (4).
    int n_restarts,
    // Gumbel noise scale used by the quality flavor. Multiplied by
    // 1/weight internally (the natural unit of Leiden gain). Lower =
    // greedier, closer to deterministic; higher = more random. A
    // negative value means "use default" (0.5).
    double temperature
);

}  // extern "C"

#endif // LEIDEN_H