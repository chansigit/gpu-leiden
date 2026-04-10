
#include <cuda_runtime.h>
#include "leiden.h"
#include <iostream>
#include <stdio.h>
#include <cstring>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iomanip>

using namespace std;

# define cuCALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort) exit(code);
   }
}


__device__ double find_to_own(Leiden_Partition& d_p, graph& d_g, double dncomm,
                              int i, int community, int comm, int chk)
{
    // Iterate over outgoing edges of node i
    for (int neighbour = d_g.out_col[i]; neighbour < d_g.out_col[i + 1]; neighbour++) {
        int target = d_g.child_out[neighbour];
        if (i != target && d_p.node_comm[target] == comm) {
            dncomm += d_g.wts_out[neighbour];
        }
    }

    // Iterate over incoming edges of node i
    for (int neighbor = d_g.in_col[i]; neighbor < d_g.in_col[i + 1]; neighbor++) {
        int target = d_g.child_in[neighbor];
        if (i != target && d_p.node_comm[target] == comm) {
            dncomm += d_g.wts_in[neighbor];
        }
    }

    return dncomm;
}


__device__ double to_community(Leiden_Partition& d_p, graph& d_g, int i, int best_comm, double dncomm)
{
    // Iterate over outgoing edges
    for (int neighbour = d_g.out_col[i]; neighbour < d_g.out_col[i + 1]; neighbour++) {
        if (d_g.child_out[neighbour] < i) {
            if (i != d_g.child_out[neighbour] && d_p.node_comm[d_g.child_out[neighbour]] == best_comm) {
                dncomm += d_g.wts_out[neighbour];
            }
        }
        else if (d_g.child_out[neighbour] > i && d_p.older_comm[d_g.child_out[neighbour]] == best_comm) {
            dncomm += d_g.wts_out[neighbour];
        }
    }

    // Iterate over incoming edges
    for (int neighbour = d_g.in_col[i]; neighbour < d_g.in_col[i + 1]; neighbour++) {
        if (d_g.child_in[neighbour] < i) {
            if (i != d_g.child_in[neighbour] && d_p.node_comm[d_g.child_in[neighbour]] == best_comm) {
                dncomm += d_g.wts_in[neighbour];
            }
        }
        else if (d_g.child_in[neighbour] > i && d_p.older_comm[d_g.child_in[neighbour]] == best_comm) {
            dncomm += d_g.wts_in[neighbour];
        }
    }

    return dncomm;
}


__device__ double removal(Leiden_Partition& d_p, graph& d_g, double dnc, int i, int comm)
{
    // Iterate over outgoing edges
    for (int neighbour = d_g.out_col[i]; neighbour < d_g.out_col[i + 1]; neighbour++) {
        if (i != d_g.child_out[neighbour]) {
            if (d_g.child_out[neighbour] < i && d_p.node_comm[d_g.child_out[neighbour]] == comm) {
                dnc += d_g.wts_out[neighbour];
            }
            else if (d_g.child_out[neighbour] > i && d_p.older_comm[d_g.child_out[neighbour]] == comm) {
                dnc += d_g.wts_out[neighbour];
            }
        }
    }

    // Iterate over incoming edges
    for (int neighbour = d_g.in_col[i]; neighbour < d_g.in_col[i + 1]; neighbour++) {
        if (i != d_g.child_in[neighbour]) {
            if (d_g.child_in[neighbour] < i && d_p.node_comm[d_g.child_in[neighbour]] == comm) {
                dnc += d_g.wts_in[neighbour];
            }
            else if (d_g.child_in[neighbour] > i && d_p.older_comm[d_g.child_in[neighbour]] == comm) {
                dnc += d_g.wts_in[neighbour];
            }
        }
    }

    return dnc;
}


__device__ int update_weights(Leiden_Partition& d_p, graph& d_g, int i, double dnc)
{
    if (d_p.node_comm[i] != d_p.older_comm[i]) {
        // Update sum_in for old and new communities using atomic operations
        atomicAdd(&d_p.sum_in[d_p.older_comm[i]], -(dnc + d_p.self_loops[i]));  // Remove contribution from old community
        atomicAdd(&d_p.sum_in[d_p.node_comm[i]], d_p.home_comm[i] + d_p.self_loops[i]); // Add contribution to new community
    }

    // Update total degrees for old and new communities
    atomicAdd(&d_p.tot_in[d_p.older_comm[i]],  -d_p.in_deg[i]);
    atomicAdd(&d_p.tot_out[d_p.older_comm[i]], -d_p.out_deg[i]);
    atomicAdd(&d_p.tot_in[d_p.node_comm[i]],    d_p.in_deg[i]);
    atomicAdd(&d_p.tot_out[d_p.node_comm[i]],   d_p.out_deg[i]);

    return 0;
}


__global__ void update_partition(Leiden_Partition d_p, graph d_g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < d_g.nodes)
    {
        int best_comm = d_p.final_comm[i];
        d_p.node_comm[i] = best_comm;

        double dncomm = 0.0;

        // Compute weight to the new community
        dncomm = to_community(d_p, d_g, i, best_comm, dncomm);

        if (best_comm != d_p.older_comm[i])
        {
            atomicAdd(&d_p.home_comm[i], dncomm);
        }

        double dnc = 0.0;

        // Compute weight removal from old community
        dnc = removal(d_p, d_g, dnc, i, d_p.older_comm[i]);

        // Update community weights
        update_weights(d_p, d_g, i, dnc);
    }
}

// Phase 0: refresh older_comm and clear home_comm before each iteration
// This was formerly done at the start of find_community, but must be a
// separate kernel so that other threads' reads of older_comm see the
// finalized values (not a partial in-progress state).
__global__ void prepare_iteration_kernel(
    int* older_comm,
    const int* node_comm,
    double* home_comm,
    int V)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        older_comm[i] = node_comm[i];
        home_comm[i] = 0.0;
    }
}

// Phase 1: For each node, compute the best candidate community and write
// final_comm[i] ONCE at the very end. All cross-thread memory dependencies
// are read-only within this kernel (node_comm, older_comm, tot_in, tot_out,
// in_deg, out_deg, self_loops, edge arrays, resolution, weight, nbrs, pos).
//
// Optimisation: single-pass edge accumulation. The OLD find_community called
// find_to_own once per candidate community, scanning ALL edges each time
// (O(degree * candidates)). This version scans edges ONCE and dispatches
// each edge to the matching candidate slot via a linear scan over a small
// local array (O(degree + candidates*avg_search)).
__global__ void find_community_phase1(Leiden_Partition d_p, graph d_g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_g.nodes) return;

    int old_comm = d_p.node_comm[i];   // equals older_comm[i] after prepare kernel
    int cand_start = d_p.pos[i];
    int cand_end   = d_p.pos[i + 1];
    int num_cands  = cand_end - cand_start;

    if (num_cands <= 0) {
        d_p.final_comm[i] = old_comm;
        return;
    }

    double inv_weight  = 1.0 / d_p.weight;
    double inv_weight2 = inv_weight * inv_weight;

    int best_comm = old_comm;
    double bestGain = 0.0;

    const int MAX_LOCAL_CANDS = 32;

    if (num_cands <= MAX_LOCAL_CANDS) {
        // --- Fast path: single-pass edge accumulation via local arrays ---
        int cand_comm[MAX_LOCAL_CANDS];
        double cand_weight[MAX_LOCAL_CANDS];

        // Load candidate community IDs (uses older_comm, same as the
        // original find_community behaviour).
        for (int c = 0; c < num_cands; c++) {
            cand_comm[c]   = d_p.older_comm[d_p.nbrs[cand_start + c]];
            cand_weight[c] = 0.0;
        }

        // Single pass over outgoing edges
        for (int e = d_g.out_col[i]; e < d_g.out_col[i + 1]; e++) {
            int target = d_g.child_out[e];
            if (target == i) continue;
            int target_comm = d_p.node_comm[target];
            double w = d_g.wts_out[e];
            for (int c = 0; c < num_cands; c++) {
                if (cand_comm[c] == target_comm) {
                    cand_weight[c] += w;
                    break;
                }
            }
        }

        // Single pass over incoming edges
        for (int e = d_g.in_col[i]; e < d_g.in_col[i + 1]; e++) {
            int target = d_g.child_in[e];
            if (target == i) continue;
            int target_comm = d_p.node_comm[target];
            double w = d_g.wts_in[e];
            for (int c = 0; c < num_cands; c++) {
                if (cand_comm[c] == target_comm) {
                    cand_weight[c] += w;
                    break;
                }
            }
        }

        // Find best community from accumulated weights
        for (int c = 0; c < num_cands; c++) {
            int comm     = cand_comm[c];
            double dncomm = cand_weight[c];

            double toc_in, toc_out;
            if (old_comm == comm) {
                toc_in  = d_p.tot_in[comm]  - d_p.in_deg[i];
                toc_out = d_p.tot_out[comm] - d_p.out_deg[i];
            } else {
                toc_in  = d_p.tot_in[comm];
                toc_out = d_p.tot_out[comm];
            }

            double newGain = (dncomm + d_p.self_loops[i]) * inv_weight
                           - d_p.resolution * (toc_in * d_p.out_deg[i] + toc_out * d_p.in_deg[i]) * inv_weight2;

            if (newGain > bestGain) {
                bestGain = newGain;
                best_comm = comm;
            }
        }

    } else {
        // --- Fallback: multi-pass for very high-degree candidate lists ---
        // Still race-free (final_comm is not written until end of kernel).
        for (int community = cand_start; community < cand_end; community++) {
            int comm = d_p.older_comm[d_p.nbrs[community]];
            double dncomm = 0.0;

            // Inlined find_to_own (same match semantics as the original:
            // node_comm[target] for current-pass matches)
            for (int e = d_g.out_col[i]; e < d_g.out_col[i + 1]; e++) {
                int target = d_g.child_out[e];
                if (target != i && d_p.node_comm[target] == comm) {
                    dncomm += d_g.wts_out[e];
                }
            }
            for (int e = d_g.in_col[i]; e < d_g.in_col[i + 1]; e++) {
                int target = d_g.child_in[e];
                if (target != i && d_p.node_comm[target] == comm) {
                    dncomm += d_g.wts_in[e];
                }
            }

            double toc_in, toc_out;
            if (old_comm == comm) {
                toc_in  = d_p.tot_in[comm]  - d_p.in_deg[i];
                toc_out = d_p.tot_out[comm] - d_p.out_deg[i];
            } else {
                toc_in  = d_p.tot_in[comm];
                toc_out = d_p.tot_out[comm];
            }

            double newGain = (dncomm + d_p.self_loops[i]) * inv_weight
                           - d_p.resolution * (toc_in * d_p.out_deg[i] + toc_out * d_p.in_deg[i]) * inv_weight2;

            if (newGain > bestGain) {
                bestGain = newGain;
                best_comm = comm;
            }
        }
    }

    // Write final_comm exactly ONCE, at the very end of the kernel.
    // Phase 2 (next kernel launch) will read it after cudaDeviceSynchronize.
    d_p.final_comm[i] = best_comm;
}

// Phase 2: Cross-node swap prevention + community size update.
// Reads d_p.final_comm (fully written by phase1), d_p.older_comm, d_p.size.
// Writes d_p.final_comm (possibly reverts), d_p.size (atomic).
__global__ void find_community_phase2(Leiden_Partition d_p, graph d_g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_g.nodes) return;

    int my_older = d_p.older_comm[i];
    int my_final = d_p.final_comm[i];

    // Prevent swaps: if i wants to move to j<i, and j wants to move to
    // i's old community, stay put.
    if (my_final < my_older && d_p.final_comm[my_final] == my_older) {
        my_final = my_older;
        d_p.final_comm[i] = my_older;
    }

    // Size bias (same as the original — prefer staying in the larger community)
    if (d_p.size[my_older] > d_p.size[my_final] && d_p.size[my_final] < d_p.size[my_older]) {
        my_final = my_older;
        d_p.final_comm[i] = my_older;
    }

    // Update sizes atomically if community actually changed
    if (my_final != my_older) {
        atomicSub(&d_p.size[my_older], 1);
        atomicAdd(&d_p.size[my_final], 1);
    }
}

 
double find_quality_cpu(Leiden_Partition& p, graph& g)
{
    double q = 0.0;

    for (int i = 0; i < g.nodes; i++)
    {
        if (p.tot_in[i] > 0 || p.tot_out[i] > 0)
        {
            q += p.sum_in[i] - p.resolution * (p.tot_in[i] * p.tot_out[i] / p.weight);
        }
    }

    q = q / p.weight;
    return q;
}

struct QualityFunctor {
    const double* sum_in;
    const double* tot_in;
    const double* tot_out;
    double inv_weight;  // precomputed 1.0/weight
    double resolution;

    __host__ __device__
    QualityFunctor(const double* si, const double* ti, const double* to, double w, double r)
        : sum_in(si), tot_in(ti), tot_out(to), inv_weight(1.0 / w), resolution(r) {}

    __host__ __device__
    double operator()(int i) const {
        if (tot_in[i] > 0 || tot_out[i] > 0) {
            return sum_in[i] - resolution * (tot_in[i] * tot_out[i] * inv_weight);
        }
        return 0.0;
    }
};

double find_quality_gpu(Leiden_Partition& d_p, int V, double weight, double resolution)
{
    thrust::counting_iterator<int> begin(0);
    thrust::counting_iterator<int> end(V);
    QualityFunctor functor(d_p.sum_in, d_p.tot_in, d_p.tot_out, weight, resolution);
    double q = thrust::transform_reduce(thrust::device, begin, end, functor, 0.0, thrust::plus<double>());
    return q / weight;
}

__global__ void count_moves_kernel(const int* node_comm, const int* older_comm, int* move_count, int V)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        if (node_comm[i] != older_comm[i]) {
            atomicAdd(move_count, 1);
        }
    }
}

// =============================================================================
// GPU graph aggregation helpers (Phase 2.2)
// =============================================================================

// Functor: pack a (src, dst) pair into a composite int64 key = src * K + dst.
// Used for thrust::transform over zip(src, dst) -> key.
struct MakeCompositeKey {
    int K;
    __host__ __device__
    int64_t operator()(const thrust::tuple<int, int>& t) const {
        return (int64_t)thrust::get<0>(t) * (int64_t)K + (int64_t)thrust::get<1>(t);
    }
};

// Functor: split a composite int64 key back into (row, col) pair.
struct SplitCompositeKey {
    int K;
    __host__ __device__
    thrust::tuple<int, int> operator()(int64_t k) const {
        int row = (int)(k / (int64_t)K);
        int col = (int)(k - (int64_t)row * (int64_t)K);
        return thrust::make_tuple(row, col);
    }
};

// Kernel: build per-edge (new_src_comm, new_dst_comm, weight) triples
// from the old CSR + the per-node new-community mapping.
// One thread per source node; thread iterates the node's out-edges.
__global__ void build_edge_triples_out(
    const int*    out_col,
    int           V,
    const int*    child_out,
    const double* wts_out,
    const int*    new_comm,
    int*          src_new,
    int*          dst_new,
    double*       wts_new)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V) return;
    int s = new_comm[i];
    int e_end = out_col[i + 1];
    for (int e = out_col[i]; e < e_end; e++) {
        int t = child_out[e];
        src_new[e] = s;
        dst_new[e] = new_comm[t];
        wts_new[e] = wts_out[e];
    }
}

int renumber_communities(Leiden_Partition& p, graph& g,
                         int* tracked_labels, int n_original)
{
    const int V_old = g.nodes;
    const int E_old = g.ed;

    auto start_time = std::chrono::high_resolution_clock::now();

    // ---------------------------------------------------------------
    // Edge case: empty or single-node graph. Nothing to aggregate.
    // ---------------------------------------------------------------
    if (V_old <= 1 || E_old == 0) {
        std::vector<int> sorted_comm(p.node_comm, p.node_comm + V_old);
        std::sort(sorted_comm.begin(), sorted_comm.end());
        sorted_comm.erase(std::unique(sorted_comm.begin(), sorted_comm.end()),
                          sorted_comm.end());
        if (tracked_labels != NULL) {
            for (int i = 0; i < n_original; i++) {
                int old_id = tracked_labels[i];
                int new_id = (int)(std::lower_bound(sorted_comm.begin(),
                                                    sorted_comm.end(),
                                                    old_id) - sorted_comm.begin());
                tracked_labels[i] = new_id;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
        cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
        cout << "Aggregate step on Device completed in " << duration.count() << " minutes!" << std::endl;
        cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
        cout << "____________________________________________" << endl;
        cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
        cout << "Preprocessing on Host completed in 0 minutes!" << std::endl;
        cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
        cout << "____________________________________________" << endl;
        return 0;
    }

    // ---------------------------------------------------------------
    // Stage A: upload p.node_comm and densify community IDs on the GPU
    // ---------------------------------------------------------------
    thrust::device_vector<int> d_node_comm(p.node_comm, p.node_comm + V_old);

    // Sort+unique of a copy gives the sorted array of distinct old community IDs.
    thrust::device_vector<int> d_unique_comms = d_node_comm;
    thrust::sort(d_unique_comms.begin(), d_unique_comms.end());
    auto unique_end = thrust::unique(d_unique_comms.begin(), d_unique_comms.end());
    int K = (int)(unique_end - d_unique_comms.begin());
    d_unique_comms.resize(K);

    // For each node, its new (dense) community id is lower_bound(unique, old_id).
    thrust::device_vector<int> d_new_comm(V_old);
    thrust::lower_bound(d_unique_comms.begin(), d_unique_comms.end(),
                        d_node_comm.begin(), d_node_comm.end(),
                        d_new_comm.begin());

    // ---------------------------------------------------------------
    // Stage B: relabel tracked_labels on the host
    // ---------------------------------------------------------------
    std::vector<int> h_unique_comms(K);
    thrust::copy(d_unique_comms.begin(), d_unique_comms.end(), h_unique_comms.begin());

    if (tracked_labels != NULL) {
        for (int i = 0; i < n_original; i++) {
            int old_id = tracked_labels[i];
            int new_id = (int)(std::lower_bound(h_unique_comms.begin(),
                                                h_unique_comms.end(),
                                                old_id) - h_unique_comms.begin());
            tracked_labels[i] = new_id;
        }
    }

    // ---------------------------------------------------------------
    // Stage C: upload old out-CSR to the device
    // ---------------------------------------------------------------
    thrust::device_vector<int>    d_out_col(g.out_col, g.out_col + V_old + 1);
    thrust::device_vector<int>    d_child_out(g.child_out, g.child_out + E_old);
    thrust::device_vector<double> d_wts_out(g.wts_out, g.wts_out + E_old);

    // ---------------------------------------------------------------
    // Stage D: build per-edge (new_src, new_dst, weight) triples
    // ---------------------------------------------------------------
    thrust::device_vector<int>    d_src_new(E_old);
    thrust::device_vector<int>    d_dst_new(E_old);
    thrust::device_vector<double> d_wts_new(E_old);

    {
        int tpb = 256;
        int nbl = (V_old + tpb - 1) / tpb;
        build_edge_triples_out<<<nbl, tpb>>>(
            thrust::raw_pointer_cast(d_out_col.data()),
            V_old,
            thrust::raw_pointer_cast(d_child_out.data()),
            thrust::raw_pointer_cast(d_wts_out.data()),
            thrust::raw_pointer_cast(d_new_comm.data()),
            thrust::raw_pointer_cast(d_src_new.data()),
            thrust::raw_pointer_cast(d_dst_new.data()),
            thrust::raw_pointer_cast(d_wts_new.data()));
        cudaDeviceSynchronize();
    }

    // Release sources we no longer need before allocating the sort buffers.
    d_out_col.clear();     d_out_col.shrink_to_fit();
    d_child_out.clear();   d_child_out.shrink_to_fit();
    d_wts_out.clear();     d_wts_out.shrink_to_fit();
    d_node_comm.clear();   d_node_comm.shrink_to_fit();
    d_unique_comms.clear(); d_unique_comms.shrink_to_fit();

    // ---------------------------------------------------------------
    // Stage E: sort by (src, dst) composite key; reduce_by_key on weights
    // ---------------------------------------------------------------
    thrust::device_vector<int64_t> d_key(E_old);
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_src_new.begin(), d_dst_new.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_src_new.end(),   d_dst_new.end())),
        d_key.begin(),
        MakeCompositeKey{K});

    // The individual src/dst arrays are no longer needed; the composite key
    // carries both. Freeing here keeps peak memory lower during the sort.
    d_src_new.clear(); d_src_new.shrink_to_fit();
    d_dst_new.clear(); d_dst_new.shrink_to_fit();

    thrust::sort_by_key(d_key.begin(), d_key.end(), d_wts_new.begin());

    // Reduce runs of identical keys: sum the weights.
    thrust::device_vector<int64_t> d_key_reduced(E_old);
    thrust::device_vector<double>  d_wts_reduced(E_old);
    auto end_pair = thrust::reduce_by_key(
        d_key.begin(), d_key.end(),
        d_wts_new.begin(),
        d_key_reduced.begin(),
        d_wts_reduced.begin());
    int E_new = (int)(end_pair.first - d_key_reduced.begin());
    d_key_reduced.resize(E_new);
    d_wts_reduced.resize(E_new);

    // Free inputs to the reduction.
    d_key.clear();     d_key.shrink_to_fit();
    d_wts_new.clear(); d_wts_new.shrink_to_fit();

    // ---------------------------------------------------------------
    // Stage F: split key back to (new_src, new_dst) and build out-CSR
    // ---------------------------------------------------------------
    thrust::device_vector<int> d_new_src(E_new);
    thrust::device_vector<int> d_new_dst(E_new);
    thrust::transform(
        d_key_reduced.begin(), d_key_reduced.end(),
        thrust::make_zip_iterator(thrust::make_tuple(d_new_src.begin(), d_new_dst.begin())),
        SplitCompositeKey{K});

    d_key_reduced.clear(); d_key_reduced.shrink_to_fit();

    // Build CSR row pointers: for each row r in [0, K], out_col[r] = index
    // of the first edge whose src >= r. This is lower_bound over the sorted
    // new_src array, indexed by a counting iterator [0, K].
    thrust::device_vector<int> d_out_col_new(K + 1);
    thrust::counting_iterator<int> row_iter(0);
    thrust::lower_bound(
        d_new_src.begin(), d_new_src.end(),
        row_iter, row_iter + K + 1,
        d_out_col_new.begin());

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    cout << "Aggregate step on Device completed in " << duration.count() << " minutes!" << std::endl;
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    cout << "____________________________________________" << endl;

    auto start_time2 = std::chrono::high_resolution_clock::now();

    // ---------------------------------------------------------------
    // Stage G: build in-CSR by re-sorting the reduced edge list
    //          with the composite key inverted to (dst * K + src).
    // ---------------------------------------------------------------
    thrust::device_vector<int64_t> d_key_in(E_new);
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_new_dst.begin(), d_new_src.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_new_dst.end(),   d_new_src.end())),
        d_key_in.begin(),
        MakeCompositeKey{K});

    // Weights follow the key permutation; make a working copy so the
    // already-ordered out-CSR weights (d_wts_reduced) stay intact.
    thrust::device_vector<double> d_wts_in(d_wts_reduced);
    thrust::sort_by_key(d_key_in.begin(), d_key_in.end(), d_wts_in.begin());

    // Split the permuted key back into (row=dst, col=src) arrays.
    thrust::device_vector<int> d_in_rows(E_new);
    thrust::device_vector<int> d_in_cols(E_new);
    thrust::transform(
        d_key_in.begin(), d_key_in.end(),
        thrust::make_zip_iterator(thrust::make_tuple(d_in_rows.begin(), d_in_cols.begin())),
        SplitCompositeKey{K});

    d_key_in.clear(); d_key_in.shrink_to_fit();

    thrust::device_vector<int> d_in_col_new(K + 1);
    thrust::counting_iterator<int> row_iter2(0);
    thrust::lower_bound(
        d_in_rows.begin(), d_in_rows.end(),
        row_iter2, row_iter2 + K + 1,
        d_in_col_new.begin());

    d_in_rows.clear(); d_in_rows.shrink_to_fit();

    // ---------------------------------------------------------------
    // Stage H: copy results back to the host graph.
    // ---------------------------------------------------------------
    // Free old host-side CSR arrays (allocated with new[] by graph_process
    // or a previous renumber_communities call).
    delete[] g.out_col;
    delete[] g.in_col;
    delete[] g.child_out;
    delete[] g.child_in;
    delete[] g.wts_out;
    delete[] g.wts_in;

    g.nodes     = K;
    g.ed        = E_new;
    g.out_col   = new int[K + 1];
    g.in_col    = new int[K + 1];
    g.child_out = new int[E_new];
    g.child_in  = new int[E_new];
    g.wts_out   = new double[E_new];
    g.wts_in    = new double[E_new];

    thrust::copy(d_out_col_new.begin(), d_out_col_new.end(), g.out_col);
    thrust::copy(d_new_dst.begin(),     d_new_dst.end(),     g.child_out);
    thrust::copy(d_wts_reduced.begin(), d_wts_reduced.end(), g.wts_out);

    thrust::copy(d_in_col_new.begin(),  d_in_col_new.end(),  g.in_col);
    thrust::copy(d_in_cols.begin(),     d_in_cols.end(),     g.child_in);
    thrust::copy(d_wts_in.begin(),      d_wts_in.end(),      g.wts_in);

    // Release the remaining device buffers explicitly before create_partition
    // allocates more host-side arrays.
    d_out_col_new.clear(); d_out_col_new.shrink_to_fit();
    d_in_col_new.clear();  d_in_col_new.shrink_to_fit();
    d_new_src.clear();     d_new_src.shrink_to_fit();
    d_new_dst.clear();     d_new_dst.shrink_to_fit();
    d_wts_reduced.clear(); d_wts_reduced.shrink_to_fit();
    d_in_cols.clear();     d_in_cols.shrink_to_fit();
    d_wts_in.clear();      d_wts_in.shrink_to_fit();

    // ---------------------------------------------------------------
    // Stage I: initialise the partition for the aggregated graph and recurse.
    // ---------------------------------------------------------------
    create_partition(g, p);

    auto end_time2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::minutes>(end_time2 - start_time2);
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    cout << "Preprocessing on Host completed in " << duration2.count() << " minutes!" << std::endl;
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    cout << "____________________________________________" << endl;

    Leiden_GPU(p, g, g.ed, tracked_labels, n_original);

    return 0;
}


Leiden_Partition create_partition(graph& g, Leiden_Partition& p)
{  
    // Allocate arrays
    p.in_deg      = new double[g.nodes];
    p.out_deg     = new double[g.nodes];
    p.tot_in      = new double[g.nodes];
    p.tot_out     = new double[g.nodes];
    p.sum_in      = new double[g.nodes];
    p.size        = new int[g.nodes];
    p.sum_kin     = new double[g.nodes];
    p.self_loops  = new double[g.nodes];
    p.node_comm   = new int[g.nodes];
    p.home_comm   = new double[g.nodes];
    p.older_comm  = new int[g.nodes];
    p.final_comm  = new int[g.nodes];
    p.nbrs        = new int[g.nodes + g.ed];
    p.pos         = new int[g.nodes + 1];

    // Initialize arrays
    for (int i = 0; i < g.nodes; i++)
    { 
        p.node_comm[i] = i;
        p.final_comm[i] = i;
        p.size[i] = 1;
        p.home_comm[i] = 0;
        p.older_comm[i] = i;
        p.in_deg[i] = 0;
        p.out_deg[i] = 0;
        p.tot_in[i] = 0;
        p.tot_out[i] = 0;
        p.sum_kin[i] = 0;
        p.self_loops[i] = 0;
        p.sum_in[i] = 0;
        p.weight = 0;
    }

    for (int i = 0; i < (g.nodes + g.ed); i++)
    { 
        p.nbrs[i] = 0;
    }

    for (int i = 0; i < (g.nodes + 1); i++)
    { 
        p.pos[i] = 0;
    }

 
    for (int i = 0; i < g.nodes; i++)
    { 
        p.in_deg[i] = indegree(g, p, i);
        p.out_deg[i] = outdegree(g, p, i);
        p.weight += p.out_deg[i];
        p.tot_in[i] = p.in_deg[i];
        p.tot_out[i] = p.out_deg[i];
        p.self_loops[i] = selfloop(g, p, i);
        p.sum_in[i] = p.self_loops[i];
    }


    p.count.clear();
    p.neigh_commNb.clear();
    p.neigh_pos.clear();

    int tot_inc = 0;
    p.count.push_back(0);

    for (int j = 0; j < g.nodes; j++)
    {
        int inc = 0;
        p.neigh_commNb.push_back(j);

        for (int i = g.out_col[j]; i < g.out_col[j + 1]; i++)
        {
            if (j != p.node_comm[g.child_out[i]])
            {
                p.neigh_commNb.push_back(p.node_comm[g.child_out[i]]);
                inc++;
            }
        }

        tot_inc += inc;
        p.count.push_back(tot_inc);
    }

    // Copy neighbor information
    for (int i = 0; i < p.neigh_commNb.size(); i++)
    {
        p.nbrs[i] = p.neigh_commNb[i];
    }

    p.neigh_pos.push_back(0); 
    for (int j = 1; j < g.nodes + 1; j++)
    {
        int b = p.count[j] - p.count[j - 1];
        int index = p.neigh_pos[j - 1];
        p.neigh_pos.push_back(index + b + 1);
    }

    for (int i = 0; i < p.neigh_pos.size(); i++)
    {
        p.pos[i] = p.neigh_pos[i];
    }

    return p;
}

inline double selfloop(graph& g, Leiden_Partition& p, int v)
{
    for (int j = g.out_col[v]; j < g.out_col[v + 1]; j++)
    {  
        if (v == g.child_out[j])
        {
            p.self_loops[v] += g.wts_out[j];
        }
    }  
    return p.self_loops[v];
}

inline double indegree(graph& g, Leiden_Partition& p, int v)
{
    for (int j = g.in_col[v]; j < g.in_col[v + 1]; j++)
    {
        p.in_deg[v] += g.wts_in[j];
    }  
    return p.in_deg[v];
}

inline double outdegree(graph& g, Leiden_Partition& p, int v)
{
    for (int j = g.out_col[v]; j < g.out_col[v + 1]; j++)
    {
        p.out_deg[v] += g.wts_out[j];
    }  
    return p.out_deg[v];
}

int Leiden_GPU(Leiden_Partition& p, graph& g, int E,
               int* tracked_labels, int n_original)
{
    Leiden_Partition d_p;
    graph d_g;

    d_p.weight = p.weight;
    d_p.resolution = p.resolution;
    d_g.nodes = g.nodes;
    int V = g.nodes;

    auto start_time = std::chrono::high_resolution_clock::now();
    double quality = 0.0;
    double imp = 0.0;

    // Allocate device memory
    cudaMalloc((void**)&d_p.node_comm, V * sizeof(int));
    cudaMalloc((void**)&d_p.size, V * sizeof(int));
    cudaMalloc((void**)&d_p.home_comm, V * sizeof(double));
    cudaMalloc((void**)&d_p.older_comm, V * sizeof(int));
    cudaMalloc((void**)&d_p.final_comm, V * sizeof(int)); 
    cudaMalloc((void**)&d_p.in_deg, V * sizeof(double));
    cudaMalloc((void**)&d_p.out_deg, V * sizeof(double));
    cudaMalloc((void**)&d_p.tot_in, V * sizeof(double));
    cudaMalloc((void**)&d_p.tot_out, V * sizeof(double));
    cudaMalloc((void**)&d_p.sum_in, V * sizeof(double));
    cudaMalloc((void**)&d_p.self_loops, V * sizeof(double));
    cudaMalloc((void**)&d_g.child_in, E * sizeof(int));
    cudaMalloc((void**)&d_g.child_out, E * sizeof(int));
    cudaMalloc((void**)&d_g.wts_in, E * sizeof(double));
    cudaMalloc((void**)&d_g.wts_out, E * sizeof(double));
    cudaMalloc((void**)&d_g.in_col, (V + 1) * sizeof(int));
    cudaMalloc((void**)&d_g.out_col, (V + 1) * sizeof(int));
    cudaMalloc((void**)&d_p.nbrs, (V + E) * sizeof(int));
    cudaMalloc((void**)&d_p.pos, (V + 1) * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_p.node_comm, p.node_comm, V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.size, p.size, V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.older_comm, p.older_comm, V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.home_comm, p.home_comm, V * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.final_comm, p.final_comm, V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.in_deg, p.in_deg, V * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.out_deg, p.out_deg, V * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.tot_in, p.tot_in, V * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.tot_out, p.tot_out, V * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.sum_in, p.sum_in, V * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.self_loops, p.self_loops, V * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g.child_in, g.child_in, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g.child_out, g.child_out, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g.wts_in, g.wts_in, E * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g.wts_out, g.wts_out, E * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g.in_col, g.in_col, (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g.out_col, g.out_col, (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.nbrs, p.nbrs, (V + E) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p.pos, p.pos, (V + 1) * sizeof(int), cudaMemcpyHostToDevice);

    int tpb = 512;
    int nbl = (g.nodes + tpb - 1) / tpb;

    int moves = 0;
    double prev_quality = 0.0;
    double q_prev_it = 0;
    quality = find_quality_gpu(d_p, V, p.weight, p.resolution);
    q_prev_it = quality;
    printf("previous quality: %f\n", q_prev_it);

    // Allocate device counter for move counting
    int* d_move_count;
    cudaMalloc((void**)&d_move_count, sizeof(int));

    // Main Leiden iteration loop
    do
    {
        moves = 0;
        prev_quality = quality;

        // Phase 0: refresh older_comm and clear home_comm (race-free setup)
        prepare_iteration_kernel <<< nbl, tpb >>>(
            d_p.older_comm, d_p.node_comm, d_p.home_comm, V);
        cudaDeviceSynchronize();

        // Phase 1: pick best community per node (single-pass edge accumulation)
        find_community_phase1 <<< nbl, tpb >>>(d_p, d_g);
        cudaDeviceSynchronize();

        // Phase 2: cross-node swap check + size update
        find_community_phase2 <<< nbl, tpb >>>(d_p, d_g);
        cudaDeviceSynchronize();

        update_partition <<< nbl, tpb >>>(d_p, d_g);
        cudaDeviceSynchronize();

        // Count moves on GPU (only 4 bytes copied back)
        cudaMemset(d_move_count, 0, sizeof(int));
        count_moves_kernel<<< nbl, tpb >>>(d_p.node_comm, d_p.older_comm, d_move_count, V);
        cudaDeviceSynchronize();
        cudaMemcpy(&moves, d_move_count, sizeof(int), cudaMemcpyDeviceToHost);

        // Compute quality on GPU
        quality = find_quality_gpu(d_p, V, d_p.weight, d_p.resolution);
        imp = quality - prev_quality;
        printf("new quality: %.6f  imp = %.6f\n", quality, imp);

    } while (moves > 0 && imp > 0.005);

    cudaFree(d_move_count);

    // Copy data back from GPU ONCE after convergence (needed for aggregation phase)
    cudaMemcpy(p.node_comm, d_p.node_comm, V * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(p.sum_in, d_p.sum_in, V * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(p.tot_in, d_p.tot_in, V * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(p.tot_out, d_p.tot_out, V * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(p.older_comm, d_p.older_comm, V * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_p.node_comm);
    cudaFree(d_p.size);
    cudaFree(d_p.home_comm);
    cudaFree(d_p.older_comm);
    cudaFree(d_p.final_comm);
    cudaFree(d_p.in_deg);
    cudaFree(d_p.out_deg);
    cudaFree(d_p.tot_in);
    cudaFree(d_p.tot_out);
    cudaFree(d_p.sum_in);
    cudaFree(d_p.self_loops);
    cudaFree(d_g.child_in);
    cudaFree(d_g.child_out);
    cudaFree(d_g.wts_in);
    cudaFree(d_g.wts_out);
    cudaFree(d_g.in_col);
    cudaFree(d_g.out_col);
    cudaFree(d_p.nbrs);
    cudaFree(d_p.pos);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    std::cout << "Leiden step completed in " << duration.count() << " minutes!" << std::endl; 
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    cout << "____________________________________________" << endl;

    if (quality > q_prev_it)
    {
        // Compose tracked_labels with p.node_comm BEFORE aggregation
        // tracked_labels[i] currently = super-node ID at this level for cell i
        // After composition: tracked_labels[i] = community ID at this level for cell i
        if (tracked_labels != NULL) {
            for (int i = 0; i < n_original; i++) {
                tracked_labels[i] = p.node_comm[tracked_labels[i]];
            }
        }
        renumber_communities(p, g, tracked_labels, n_original);
    }
    else
    {
        // Final level reached: compose once to get final labels
        // After this, tracked_labels[i] = final community ID for cell i
        if (tracked_labels != NULL) {
            for (int i = 0; i < n_original; i++) {
                tracked_labels[i] = p.node_comm[tracked_labels[i]];
            }
        }
        cout << "Leiden_GPU done and dusted :)" << endl;
    }

    return 0;
}

// ============================================================
// C API entry point for Python/external callers
// Accepts CSR arrays directly - no file I/O needed
// ============================================================

extern "C" {

int leiden_from_csr(
    // Out-edges CSR (for undirected/symmetric input, same as in-edges CSR)
    const int*    out_indptr,   // [n_nodes + 1]
    const int*    out_indices,  // [n_out_edges]
    const double* out_data,     // [n_out_edges]
    int n_out_edges,
    // In-edges CSR
    const int*    in_indptr,    // [n_nodes + 1]
    const int*    in_indices,   // [n_in_edges]
    const double* in_data,      // [n_in_edges]
    int n_in_edges,
    // Graph size
    int n_nodes,
    // Algorithm parameters
    double resolution,
    int max_iterations,         // -1 = unlimited (currently ignored, always runs to convergence)
    unsigned int random_seed,   // currently ignored
    // Output (caller-allocated)
    int* out_labels             // [n_nodes] - filled with community ID per node
)
{
    // Build host-side graph struct (CSR format)
    // NOTE: graph uses int / double which matches scipy's default int32 / float64
    graph g;
    g.nodes = n_nodes;
    g.ed = n_out_edges;   // legacy field used by create_c_partition for nbrs/pos allocation

    g.out_col = new int[n_nodes + 1];
    g.in_col  = new int[n_nodes + 1];
    g.child_out = new int[n_out_edges];
    g.child_in  = new int[n_in_edges];
    g.wts_out = new double[n_out_edges];
    g.wts_in  = new double[n_in_edges];

    std::memcpy(g.out_col,   out_indptr,  (n_nodes + 1) * sizeof(int));
    std::memcpy(g.in_col,    in_indptr,   (n_nodes + 1) * sizeof(int));
    std::memcpy(g.child_out, out_indices, n_out_edges * sizeof(int));
    std::memcpy(g.child_in,  in_indices,  n_in_edges  * sizeof(int));
    std::memcpy(g.wts_out,   out_data,    n_out_edges * sizeof(double));
    std::memcpy(g.wts_in,    in_data,     n_in_edges  * sizeof(double));

    // Build partition (reuses existing CPU-side initialization)
    Leiden_Partition p;
    p.resolution = resolution;
    create_c_partition(g, p);

    // Allocate label tracker and initialize to identity (each cell is its own super-node initially)
    int* tracked_labels = new int[n_nodes];
    for (int i = 0; i < n_nodes; i++) {
        tracked_labels[i] = i;
    }

    // Run Leiden with label tracking
    // Note: Leiden_GPU may modify g (aggregation overwrites it), but tracked_labels is maintained
    // across recursion levels so at return, tracked_labels[i] = final community for original cell i.
    Leiden_GPU(p, g, n_out_edges, tracked_labels, n_nodes);

    // Copy final labels to caller's buffer
    std::memcpy(out_labels, tracked_labels, n_nodes * sizeof(int));

    // Cleanup
    delete[] tracked_labels;
    free(g);         // project-local free(graph&) from leiden.cpp
    free_part(p);    // from leiden.cpp

    return 0;
}

}  // extern "C"
