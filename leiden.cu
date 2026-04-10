
#include <cuda_runtime.h>
#include <curand_kernel.h>
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
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iomanip>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

// Global verbosity flag (see declaration in leiden.h). Default 1 so
// that CLI-mode runs keep the output tests/verify.sh depends on.
// `leiden_from_csr` overrides from its `verbose` parameter on entry.
int gpu_leiden_verbose = 1;

# define cuCALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort) exit(code);
   }
}


// Phase 2.5b: deterministic move + community-stats recompute.
//
// The former update_partition kernel (plus its helpers to_community,
// removal, update_weights, find_to_own) committed node moves and
// incrementally updated tot_in/tot_out/sum_in via atomicAdd on shared
// community slots. Floating-point atomicAdd is non-associative across
// different accumulation orders, so repeated runs produced tiny numerical
// drift in the community stats and, after enough iterations, diverged
// to different cluster assignments.
//
// Replacement: a two-step sequence, both bit-deterministic.
//
//   Step A — apply_moves_assign_kernel:
//     writes d_p.node_comm[i] = d_p.final_comm[i] (only thread i writes slot i,
//     no race at all).
//
//   Step B — apply_moves_home_kernel (launched after a cudaDeviceSynchronize):
//     reads the now-finalised node_comm for every edge endpoint and computes
//     home_comm[i] = sum of weights to neighbours in the new community.
//
//   Step C — recompute_community_stats_gpu (host-side helper, below):
//     rebuilds tot_in[c], tot_out[c], and sum_in[c] from scratch using
//     thrust::sort_by_key + thrust::reduce_by_key + thrust::scatter.
//     These Thrust primitives are deterministic: sort_by_key (stable merge /
//     radix), reduce_by_key (contiguous-run folding), and scatter (distinct
//     indices, no race).
//
// Combined with the warp-shuffle phase-1 reduction, run-to-run output is
// bit-identical at the cost of an extra O(V log V) sort per outer iteration.

__global__ void apply_moves_assign_kernel(Leiden_Partition d_p, graph d_g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_g.nodes) return;
    // Pure per-thread write; no cross-thread dependency.
    d_p.node_comm[i] = d_p.final_comm[i];
}

__global__ void apply_moves_home_kernel(Leiden_Partition d_p, graph d_g)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_g.nodes) return;

    // node_comm is fully committed at this point (separate kernel launched
    // after cudaDeviceSynchronize), so every read sees the final value.
    int best_comm = d_p.node_comm[i];
    double dncomm = 0.0;

    for (int e = d_g.out_col[i]; e < d_g.out_col[i + 1]; e++) {
        int t = d_g.child_out[e];
        if (t != i && d_p.node_comm[t] == best_comm) {
            dncomm += d_g.wts_out[e];
        }
    }
    for (int e = d_g.in_col[i]; e < d_g.in_col[i + 1]; e++) {
        int t = d_g.child_in[e];
        if (t != i && d_p.node_comm[t] == best_comm) {
            dncomm += d_g.wts_in[e];
        }
    }
    d_p.home_comm[i] = dncomm;
}

// Per-node internal-edge weight for sum_in reconstruction:
//   per_node_internal[v] = self_loops[v]
//                        + sum_{out edges (v,u), u != v, node_comm[u]==node_comm[v]} wts_out
//                        + sum_{in  edges (v,u), u != v, node_comm[u]==node_comm[v]} wts_in
__global__ void compute_per_node_internal_weight_kernel(
    const int*    node_comm,
    const int*    out_col,
    const int*    child_out,
    const double* wts_out,
    const int*    in_col,
    const int*    child_in,
    const double* wts_in,
    const double* self_loops,
    double*       per_node_internal,
    int           V)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;
    int c = node_comm[v];
    double s = self_loops[v];
    for (int e = out_col[v]; e < out_col[v + 1]; e++) {
        int u = child_out[e];
        if (u != v && node_comm[u] == c) s += wts_out[e];
    }
    for (int e = in_col[v]; e < in_col[v + 1]; e++) {
        int u = child_in[e];
        if (u != v && node_comm[u] == c) s += wts_in[e];
    }
    per_node_internal[v] = s;
}

// Host-side helper: deterministically recompute tot_in[c], tot_out[c], and
// sum_in[c] for all communities c, given the current d_p.node_comm.
//
// Working buffers (d_sort_keys, d_sort_idx, d_sorted_vals, d_reduced_keys,
// d_reduced_vals, d_per_node_internal) are pre-allocated in Leiden_GPU so
// this is called once per outer iteration without malloc/free churn.
static void recompute_community_stats_gpu(
    Leiden_Partition& d_p,
    graph&            d_g,
    int               V,
    int*              d_sort_keys,
    int*              d_sort_idx,
    double*           d_sorted_vals,
    int*              d_reduced_keys,
    double*           d_reduced_vals,
    double*           d_per_node_internal)
{
    // --- 1. Compute per-node internal edge weight (for sum_in reduction). ---
    int tpb = 256;
    int nbl = (V + tpb - 1) / tpb;
    compute_per_node_internal_weight_kernel<<<nbl, tpb>>>(
        d_p.node_comm,
        d_g.out_col, d_g.child_out, d_g.wts_out,
        d_g.in_col,  d_g.child_in,  d_g.wts_in,
        d_p.self_loops,
        d_per_node_internal,
        V);
    cudaDeviceSynchronize();

    // --- 2. Sort (community, node_idx) pairs once by community. ---
    // This sorted ordering drives all three community reductions (tot_in,
    // tot_out, sum_in) via thrust::gather over d_sort_idx.
    cudaMemcpy(d_sort_keys, d_p.node_comm, V * sizeof(int),
               cudaMemcpyDeviceToDevice);
    thrust::sequence(thrust::device, d_sort_idx, d_sort_idx + V);
    thrust::sort_by_key(thrust::device,
                        d_sort_keys, d_sort_keys + V,
                        d_sort_idx);

    // --- 3. tot_in[c] = sum of in_deg[v] for v in community c. ---
    thrust::gather(thrust::device,
                   d_sort_idx, d_sort_idx + V,
                   d_p.in_deg,
                   d_sorted_vals);
    auto end_pair = thrust::reduce_by_key(
        thrust::device,
        d_sort_keys, d_sort_keys + V,
        d_sorted_vals,
        d_reduced_keys,
        d_reduced_vals);
    int K_reduced = (int)(end_pair.first - d_reduced_keys);
    cudaMemset(d_p.tot_in, 0, V * sizeof(double));
    thrust::scatter(thrust::device,
                    d_reduced_vals, d_reduced_vals + K_reduced,
                    d_reduced_keys,
                    d_p.tot_in);

    // --- 4. tot_out[c] = sum of out_deg[v] for v in community c. ---
    thrust::gather(thrust::device,
                   d_sort_idx, d_sort_idx + V,
                   d_p.out_deg,
                   d_sorted_vals);
    end_pair = thrust::reduce_by_key(
        thrust::device,
        d_sort_keys, d_sort_keys + V,
        d_sorted_vals,
        d_reduced_keys,
        d_reduced_vals);
    K_reduced = (int)(end_pair.first - d_reduced_keys);
    cudaMemset(d_p.tot_out, 0, V * sizeof(double));
    thrust::scatter(thrust::device,
                    d_reduced_vals, d_reduced_vals + K_reduced,
                    d_reduced_keys,
                    d_p.tot_out);

    // --- 5. sum_in[c] = sum of per_node_internal[v] for v in community c. ---
    thrust::gather(thrust::device,
                   d_sort_idx, d_sort_idx + V,
                   d_per_node_internal,
                   d_sorted_vals);
    end_pair = thrust::reduce_by_key(
        thrust::device,
        d_sort_keys, d_sort_keys + V,
        d_sorted_vals,
        d_reduced_keys,
        d_reduced_vals);
    K_reduced = (int)(end_pair.first - d_reduced_keys);
    cudaMemset(d_p.sum_in, 0, V * sizeof(double));
    thrust::scatter(thrust::device,
                    d_reduced_vals, d_reduced_vals + K_reduced,
                    d_reduced_keys,
                    d_p.sum_in);
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

// Phase 1 (warp-cooperative): one warp per node, edges parallelized across lanes.
//
// Each warp processes candidates in chunks of MAX_CHUNK (=32). The warp's
// shared-memory slice holds only cand_comm[MAX_CHUNK] for the current chunk
// (the per-candidate weight accumulators now live in per-lane local memory
// and are reduced via warp shuffles rather than shared-memory atomics). With
// block size 512 (16 warps), per-block shared memory = 16 * (32*4) = 2 KB.
//
// Deterministic accumulation: each lane maintains its own private
// chunk_weights[MAX_CHUNK] array (spills to per-thread local memory, but
// indexed coalescedly across lanes and L1-cached). Per candidate c in the
// chunk, a butterfly __shfl_down_sync reduction with fixed offsets
// (16,8,4,2,1) sums the 32 lanes' contributions in a fixed tree order,
// yielding bit-identical results across runs. Lane c then computes the
// gain for its candidate and updates a per-lane running best; a final
// __shfl_xor_sync butterfly reduction picks the overall warp best.
#define PHASE1_WARP_CHUNK 32

__global__ void find_community_phase1_warp(Leiden_Partition d_p, graph d_g)
{
    extern __shared__ unsigned char smem[];
    const int WARP_SIZE = 32;
    const int MAX_CHUNK = PHASE1_WARP_CHUNK;

    int warps_per_block = blockDim.x / WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id_global = blockIdx.x * warps_per_block + warp_id_in_block;
    int i = warp_id_global;  // one warp per node

    if (i >= d_g.nodes) return;

    // Per-warp shared memory slice: only cand_comm[MAX_CHUNK] (read-only
    // across the warp once the chunk header is loaded). Weights are
    // accumulated in per-lane local memory, not here.
    int* sh_cand_comm = (int*)(smem + warp_id_in_block * (MAX_CHUNK * sizeof(int)));

    int old_comm = d_p.node_comm[i];
    int cand_start = d_p.pos[i];
    int cand_end   = d_p.pos[i + 1];
    int num_cands  = cand_end - cand_start;

    if (num_cands <= 0) {
        if (lane == 0) d_p.final_comm[i] = old_comm;
        return;
    }

    double inv_w  = 1.0 / d_p.weight;
    double inv_w2 = inv_w * inv_w;
    double self_i = d_p.self_loops[i];
    double in_i   = d_p.in_deg[i];
    double out_i  = d_p.out_deg[i];
    double res    = d_p.resolution;

    int e_out_start = d_g.out_col[i];
    int e_out_end   = d_g.out_col[i + 1];
    int e_in_start  = d_g.in_col[i];
    int e_in_end    = d_g.in_col[i + 1];

    // Per-lane running best over all chunks; merged across lanes at the end.
    double running_best_gain = 0.0;
    int    running_best_comm = old_comm;

    // Process candidates in chunks of up to MAX_CHUNK (32)
    for (int chunk_start = 0; chunk_start < num_cands; chunk_start += MAX_CHUNK) {
        int chunk_end = chunk_start + MAX_CHUNK;
        if (chunk_end > num_cands) chunk_end = num_cands;
        int chunk_size = chunk_end - chunk_start;

        // Load this chunk's candidate community IDs. With MAX_CHUNK == WARP_SIZE,
        // each lane loads at most one entry.
        if (lane < chunk_size) {
            sh_cand_comm[lane] = d_p.older_comm[d_p.nbrs[cand_start + chunk_start + lane]];
        }
        __syncwarp();

        // Per-lane private accumulators for THIS chunk. Dynamic indexing
        // forces spill to local memory, which is OK: accesses are
        // coalesced across lanes and L1-cached.
        double chunk_weights[MAX_CHUNK];
        #pragma unroll
        for (int c = 0; c < MAX_CHUNK; c++) chunk_weights[c] = 0.0;

        // Outgoing edges, stride-32 across the warp. Each lane accumulates
        // into ITS own chunk_weights[] — no cross-lane race.
        for (int e = e_out_start + lane; e < e_out_end; e += WARP_SIZE) {
            int target = d_g.child_out[e];
            if (target == i) continue;
            int target_comm = d_p.node_comm[target];
            double w = d_g.wts_out[e];
            for (int c = 0; c < chunk_size; c++) {
                if (sh_cand_comm[c] == target_comm) {
                    chunk_weights[c] += w;  // lane-private, deterministic
                    break;
                }
            }
        }

        // Incoming edges
        for (int e = e_in_start + lane; e < e_in_end; e += WARP_SIZE) {
            int target = d_g.child_in[e];
            if (target == i) continue;
            int target_comm = d_p.node_comm[target];
            double w = d_g.wts_in[e];
            for (int c = 0; c < chunk_size; c++) {
                if (sh_cand_comm[c] == target_comm) {
                    chunk_weights[c] += w;  // lane-private, deterministic
                    break;
                }
            }
        }

        // Deterministic per-candidate warp reduction. For each candidate c
        // in the chunk, sum chunk_weights[c] across all 32 lanes via a
        // butterfly __shfl_down_sync with fixed offsets (16, 8, 4, 2, 1).
        // ALL lanes participate in every shuffle — putting the shuffle
        // inside an `if (lane == c)` branch would be divergent and hang.
        for (int c = 0; c < chunk_size; c++) {
            double val = chunk_weights[c];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                val += __shfl_down_sync(0xffffffff, val, off);
            }
            // After the tree reduction, lane 0 holds the total. Broadcast
            // to the owner lane (c), which updates its per-lane running best.
            double total = __shfl_sync(0xffffffff, val, 0);

            if (lane == c) {
                int comm = sh_cand_comm[c];
                double dncomm = total;

                double toc_in, toc_out;
                if (old_comm == comm) {
                    toc_in  = d_p.tot_in[comm]  - in_i;
                    toc_out = d_p.tot_out[comm] - out_i;
                } else {
                    toc_in  = d_p.tot_in[comm];
                    toc_out = d_p.tot_out[comm];
                }

                double gain = (dncomm + self_i) * inv_w
                            - res * (toc_in * out_i + toc_out * in_i) * inv_w2;

                if (gain > running_best_gain) {
                    running_best_gain = gain;
                    running_best_comm = comm;
                }
            }
        }
        __syncwarp();
    }

    // Final warp-shuffle butterfly reduction: merge the 32 per-lane
    // running bests into one warp-wide best. Lanes that were never the
    // owner of any candidate (e.g. lanes >= chunk_size on the last
    // chunk, when the node has fewer total candidates) carry the
    // default (running_best_gain == 0.0, running_best_comm == old_comm),
    // which is the correct "no-op" fallback.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other_gain = __shfl_xor_sync(0xffffffff, running_best_gain, offset);
        int    other_comm = __shfl_xor_sync(0xffffffff, running_best_comm, offset);
        if (other_gain > running_best_gain) {
            running_best_gain = other_gain;
            running_best_comm = other_comm;
        }
    }

    if (lane == 0) {
        d_p.final_comm[i] = running_best_comm;
    }
}

// Note: the warp-cooperative kernel above handles arbitrary num_cands by
// chunking the candidate array. No thread-per-node fallback is launched.

// =============================================================================
// Phase 3.1: PROBABILISTIC phase1 kernel (Gumbel-max sampling).
// =============================================================================
//
// Identical to find_community_phase1_warp above in structure, but replaces
// the greedy "pick candidate with max gain" with a softmax sample from
// P(c) ∝ exp(gain[c] / temperature) implemented via the Gumbel-max trick:
//   sampled_c = argmax_c (gain[c] + temperature * G_c),  G_c ~ Gumbel(0,1).
//
// Candidates with gain <= 0 are excluded (staying put is always an option;
// setting their score to -infinity ensures they're never sampled).
//
// Per-lane state carries (best_score, best_gain, best_comm). The final warp
// reduction picks the lane with the largest SCORE (not gain), so that the
// Gumbel sample reaches the owner of the winning candidate.
//
// Each thread gets its own curand Philox state, so different lanes in the
// same warp draw independent Gumbel samples for the candidates they own.
// Within a chunk, each owner lane calls curand_uniform() ONCE per candidate
// it owns; across chunks, the per-lane state advances so successive draws
// are independent.
__global__ void init_rng_kernel(
    curandStatePhilox4_32_10_t* states,
    unsigned long long          seed,
    int                         n_states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_states) return;
    // Each thread gets a distinct sub-sequence to guarantee independent streams.
    curand_init(seed, (unsigned long long)i, 0, &states[i]);
}

__global__ void find_community_phase1_warp_prob(
    Leiden_Partition            d_p,
    graph                       d_g,
    curandStatePhilox4_32_10_t* rng_states,
    double                      temperature)
{
    extern __shared__ unsigned char smem[];
    const int WARP_SIZE = 32;
    const int MAX_CHUNK = PHASE1_WARP_CHUNK;

    int warps_per_block  = blockDim.x / WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int lane             = threadIdx.x & (WARP_SIZE - 1);
    int warp_id_global   = blockIdx.x * warps_per_block + warp_id_in_block;
    int i                = warp_id_global;  // one warp per node

    if (i >= d_g.nodes) return;

    // Per-lane RNG state (one state per (warp, lane) = per global thread).
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t local_state = rng_states[global_thread_id];

    // Per-warp shared memory slice: only cand_comm[MAX_CHUNK] (read-only
    // across the warp once the chunk header is loaded). Weights are
    // accumulated in per-lane local memory, not here.
    int* sh_cand_comm = (int*)(smem + warp_id_in_block * (MAX_CHUNK * sizeof(int)));

    int old_comm   = d_p.node_comm[i];
    int cand_start = d_p.pos[i];
    int cand_end   = d_p.pos[i + 1];
    int num_cands  = cand_end - cand_start;

    if (num_cands <= 0) {
        if (lane == 0) d_p.final_comm[i] = old_comm;
        // Save RNG state even on early return so subsequent kernel
        // invocations pick up at the right position (not strictly needed
        // for correctness because the state was only read, but tidy).
        rng_states[global_thread_id] = local_state;
        return;
    }

    double inv_w  = 1.0 / d_p.weight;
    double inv_w2 = inv_w * inv_w;
    double self_i = d_p.self_loops[i];
    double in_i   = d_p.in_deg[i];
    double out_i  = d_p.out_deg[i];
    double res    = d_p.resolution;

    int e_out_start = d_g.out_col[i];
    int e_out_end   = d_g.out_col[i + 1];
    int e_in_start  = d_g.in_col[i];
    int e_in_end    = d_g.in_col[i + 1];

    // Per-lane running best over all chunks. We track (score, gain, comm).
    // Initial state = "stay put" with score 0, gain 0, comm = old_comm.
    // Any sampled candidate whose Gumbel-perturbed score exceeds 0
    // takes over. If all positive-gain candidates happen to draw very
    // negative Gumbel values, stay-put wins, which is the correct safe
    // fallback.
    double running_best_score = 0.0;
    double running_best_gain  = 0.0;
    int    running_best_comm  = old_comm;

    for (int chunk_start = 0; chunk_start < num_cands; chunk_start += MAX_CHUNK) {
        int chunk_end = chunk_start + MAX_CHUNK;
        if (chunk_end > num_cands) chunk_end = num_cands;
        int chunk_size = chunk_end - chunk_start;

        if (lane < chunk_size) {
            sh_cand_comm[lane] = d_p.older_comm[d_p.nbrs[cand_start + chunk_start + lane]];
        }
        __syncwarp();

        double chunk_weights[MAX_CHUNK];
        #pragma unroll
        for (int c = 0; c < MAX_CHUNK; c++) chunk_weights[c] = 0.0;

        for (int e = e_out_start + lane; e < e_out_end; e += WARP_SIZE) {
            int target = d_g.child_out[e];
            if (target == i) continue;
            int target_comm = d_p.node_comm[target];
            double w = d_g.wts_out[e];
            for (int c = 0; c < chunk_size; c++) {
                if (sh_cand_comm[c] == target_comm) {
                    chunk_weights[c] += w;
                    break;
                }
            }
        }

        for (int e = e_in_start + lane; e < e_in_end; e += WARP_SIZE) {
            int target = d_g.child_in[e];
            if (target == i) continue;
            int target_comm = d_p.node_comm[target];
            double w = d_g.wts_in[e];
            for (int c = 0; c < chunk_size; c++) {
                if (sh_cand_comm[c] == target_comm) {
                    chunk_weights[c] += w;
                    break;
                }
            }
        }

        for (int c = 0; c < chunk_size; c++) {
            double val = chunk_weights[c];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                val += __shfl_down_sync(0xffffffff, val, off);
            }
            double total = __shfl_sync(0xffffffff, val, 0);

            if (lane == c) {
                int comm = sh_cand_comm[c];
                double dncomm = total;

                double toc_in, toc_out;
                if (old_comm == comm) {
                    toc_in  = d_p.tot_in[comm]  - in_i;
                    toc_out = d_p.tot_out[comm] - out_i;
                } else {
                    toc_in  = d_p.tot_in[comm];
                    toc_out = d_p.tot_out[comm];
                }

                double gain = (dncomm + self_i) * inv_w
                            - res * (toc_in * out_i + toc_out * in_i) * inv_w2;

                // Build Gumbel-perturbed score. Candidates with gain <= 0
                // are never sampled ("stay put" is implicitly represented
                // by the initial running_best_score = 0).
                //
                //   score = gain + temperature * inv_w * G
                //
                // Scaling noise by inv_w (= 1/total_weight) ties the
                // per-move noise magnitude to the Leiden objective's
                // natural unit: typical gain = O(1/weight) on Leiden
                // graphs. A temperature of 0.2 corresponds to noise on
                // the order of 20% of the unit gain magnitude, which is
                // empirically the sweet spot for breaking ties without
                // overriding clear winners across a wide variety of
                // graph sizes.
                double score;
                if (gain > 0.0) {
                    float u = curand_uniform(&local_state);
                    if (u < 1e-20f) u = 1e-20f;
                    if (u > 1.0f - 1e-7f) u = 1.0f - 1e-7f;
                    float gumbel = -logf(-logf(u));
                    double noise_scale = temperature * inv_w;
                    score = gain + noise_scale * (double)gumbel;
                } else {
                    score = -1e300;
                }

                if (score > running_best_score) {
                    running_best_score = score;
                    running_best_gain  = gain;
                    running_best_comm  = comm;
                }
            }
        }
        __syncwarp();
    }

    // Final warp-shuffle butterfly reduction on SCORE (not gain): pick the
    // lane whose Gumbel-perturbed candidate wins. Carry (score, gain, comm)
    // through the reduction.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other_score = __shfl_xor_sync(0xffffffff, running_best_score, offset);
        double other_gain  = __shfl_xor_sync(0xffffffff, running_best_gain,  offset);
        int    other_comm  = __shfl_xor_sync(0xffffffff, running_best_comm,  offset);
        if (other_score > running_best_score) {
            running_best_score = other_score;
            running_best_gain  = other_gain;
            running_best_comm  = other_comm;
        }
    }

    if (lane == 0) {
        d_p.final_comm[i] = running_best_comm;
    }

    // Save RNG state for the next kernel invocation. Every thread writes
    // its own slot — no race.
    rng_states[global_thread_id] = local_state;
    (void)running_best_gain;  // kept for potential debug; silence unused warning
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

// Phase 2 (deterministic variant, Phase 2.5b).
//
// The original find_community_phase2 (above) has two read-modify-write
// races that make its output non-deterministic on large graphs:
//   1. It reads d_p.final_comm[my_final] AFTER other threads may have
//      written to d_p.final_comm[my_final] (when those threads revert).
//      Whether a given thread sees the original phase-1 value or the
//      reverted value depends on kernel scheduling.
//   2. It reads d_p.size[my_older] and d_p.size[my_final] concurrently
//      with other threads' atomicAdd/atomicSub updates to the same
//      community size counters; the value observed depends on scheduling.
//
// This variant eliminates both races by operating on pre-computed
// snapshots of final_comm and size (taken immediately before launch via
// cudaMemcpyAsync). The snapshots are not modified during the kernel, so
// every thread sees the same well-defined input regardless of scheduling.
// Writes to d_p.final_comm[i] are per-thread unique (no race), and the
// final size delta is accumulated into per-node (old_comm, new_comm)
// slots; the caller then applies those deltas with a separate
// deterministic Thrust reduce_by_key + scatter pass.
//
// This matches the original kernel's SEMANTICS exactly (same swap
// prevention, same size-bias tie-break) — it only changes the source
// of the reads so that they are well-defined across thread scheduling.
__global__ void find_community_phase2_det(
    Leiden_Partition d_p,
    graph            d_g,
    const int*       final_snapshot,  // snapshot of d_p.final_comm BEFORE phase2
    const int*       size_snapshot)   // snapshot of d_p.size        BEFORE phase2
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d_g.nodes) return;

    int my_older = d_p.older_comm[i];
    int my_final = final_snapshot[i];

    // Race-free swap prevention (read other thread's DECISION from snapshot).
    if (my_final < my_older && final_snapshot[my_final] == my_older) {
        my_final = my_older;
    }

    // Race-free size-bias tie-break (read size from snapshot).
    if (size_snapshot[my_older] > size_snapshot[my_final]
        && size_snapshot[my_final] < size_snapshot[my_older]) {
        my_final = my_older;
    }

    // Commit the final decision. No thread writes the same slot.
    d_p.final_comm[i] = my_final;
}

// After find_community_phase2_det has written the final decisions into
// d_p.final_comm, these two helpers rebuild d_p.size (nodes-per-community
// counts) from scratch. Clear size[] to zero, then atomicAdd +1 per node
// into size[final_comm[i]]. Integer atomicAdd is fully commutative and
// associative, so the final value is deterministic regardless of the
// order in which threads fire.
__global__ void clear_size_kernel(int* size, int V)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) size[i] = 0;
}

__global__ void accumulate_size_kernel(const int* final_comm, int* size, int V)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) {
        atomicAdd(&size[final_comm[i]], 1);
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

// Phase 3.1: flavor/seed/temperature/out_modularity are propagated to the
// recursive Leiden_GPU call on the aggregated graph so that the quality
// flavor stays active across all hierarchy levels.
int renumber_communities(Leiden_Partition& p, graph& g,
                         int* tracked_labels, int n_original,
                         int flavor, unsigned int random_seed,
                         double temperature, double* out_modularity)
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
        if (gpu_leiden_verbose) {
            cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
            cout << "Aggregate step on Device completed in " << duration.count() << " minutes!" << std::endl;
            cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
            cout << "____________________________________________" << endl;
            cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
            cout << "Preprocessing on Host completed in 0 minutes!" << std::endl;
            cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
            cout << "____________________________________________" << endl;
        }
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
    if (gpu_leiden_verbose) {
        cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
        cout << "Aggregate step on Device completed in " << duration.count() << " minutes!" << std::endl;
        cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
        cout << "____________________________________________" << endl;
    }

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
    if (gpu_leiden_verbose) {
        cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
        cout << "Preprocessing on Host completed in " << duration2.count() << " minutes!" << std::endl;
        cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
        cout << "____________________________________________" << endl;
    }

    Leiden_GPU(p, g, g.ed, tracked_labels, n_original,
               flavor, random_seed, temperature, out_modularity);

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

// Phase 3.1: Leiden_GPU now takes flavor + random_seed + temperature to
// support two co-existing paths:
//   - flavor = 0 (deterministic): bit-reproducible, v0.2 behaviour. seed +
//     temperature are ignored. out_modularity is written if non-null.
//   - flavor = 1 (quality):       probabilistic Gumbel-max in phase1 and
//     softmax sampling in the CPU refinement. seed is used to init the
//     per-thread curand state; temperature sets the softmax sharpness.
//
// out_modularity (if non-null) receives the post-convergence modularity
// of the DEEPEST level of the aggregation hierarchy. Because the
// recursion unwinds with the deepest call writing LAST, the pointer ends
// up holding the terminal-level modularity, which is exactly the final
// modularity of the hierarchical partition under the original graph.
int Leiden_GPU(Leiden_Partition& p, graph& g, int E,
               int* tracked_labels, int n_original,
               int flavor, unsigned int random_seed,
               double temperature, double* out_modularity)
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

    // Per-Leiden_GPU refinement call counter, mixed into the seed so that
    // successive calls in the same run (including recursion) use distinct
    // RNG streams.
    int refine_call_count = 0;

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

    // Phase 2.5b working buffers for deterministic community-stats recompute.
    // Allocated once, reused across every outer iteration.
    int*    d_sort_keys;
    int*    d_sort_idx;
    double* d_sorted_vals;
    int*    d_reduced_keys;
    double* d_reduced_vals;
    double* d_per_node_internal;
    cudaMalloc((void**)&d_sort_keys,         V * sizeof(int));
    cudaMalloc((void**)&d_sort_idx,          V * sizeof(int));
    cudaMalloc((void**)&d_sorted_vals,       V * sizeof(double));
    cudaMalloc((void**)&d_reduced_keys,      V * sizeof(int));
    cudaMalloc((void**)&d_reduced_vals,      V * sizeof(double));
    cudaMalloc((void**)&d_per_node_internal, V * sizeof(double));

    // Phase 2.5b snapshot buffers for deterministic phase-2 swap check.
    // find_community_phase2_det reads from these instead of the live
    // d_p.final_comm / d_p.size arrays so that its decisions are independent
    // of thread scheduling.
    int* d_final_snapshot;
    int* d_size_snapshot;
    cudaMalloc((void**)&d_final_snapshot, V * sizeof(int));
    cudaMalloc((void**)&d_size_snapshot,  V * sizeof(int));

    // Phase 3.1: per-thread curand Philox state for the quality flavor.
    // One state per thread in the warp-cooperative phase1 launch
    // (warp_blocks * tpb threads total). Allocated only when flavor==1.
    curandStatePhilox4_32_10_t* d_rng_states = nullptr;
    int rng_n_threads = 0;

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

    // Phase 3.1: init per-thread curand states for the quality flavor.
    // Must match the warp-cooperative phase1 launch configuration:
    // warps_per_block = tpb/32, warp_blocks = ceil(V / warps_per_block),
    // total threads = warp_blocks * tpb.
    if (flavor == 1) {
        int warps_per_block = tpb / 32;
        int warp_blocks = (V + warps_per_block - 1) / warps_per_block;
        rng_n_threads = warp_blocks * tpb;
        cudaMalloc((void**)&d_rng_states,
                   (size_t)rng_n_threads * sizeof(curandStatePhilox4_32_10_t));
        // Mix the caller's random_seed with the current graph size V so
        // different aggregation levels draw from different base streams.
        // (recursive Leiden_GPU calls see different V, so this implicitly
        // decorrelates the RNG at each recursion depth.)
        unsigned long long base_seed =
            (unsigned long long)random_seed * 0x9E3779B97F4A7C15ULL
            + (unsigned long long)(V + 1) * 0xDEADBEEFCAFEBABEULL;
        init_rng_kernel<<<warp_blocks, tpb>>>(d_rng_states, base_seed, rng_n_threads);
        cudaDeviceSynchronize();
    }

    int moves = 0;
    double prev_quality = 0.0;
    double q_prev_it = 0;
    quality = find_quality_gpu(d_p, V, p.weight, p.resolution);
    q_prev_it = quality;
    if (gpu_leiden_verbose) printf("previous quality: %f\n", q_prev_it);

    // Allocate device counter for move counting
    int* d_move_count;
    cudaMalloc((void**)&d_move_count, sizeof(int));

    // PROFILE: cumulative kernel timings via cudaEvent
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    float ms_prepare = 0, ms_phase1 = 0, ms_phase2 = 0, ms_update = 0, ms_count = 0, ms_quality = 0;
    int n_iters = 0;

    // Main Leiden iteration loop
    do
    {
        n_iters++;
        moves = 0;
        prev_quality = quality;

        // Phase 0: refresh older_comm and clear home_comm (race-free setup)
        cudaEventRecord(ev_start);
        prepare_iteration_kernel <<< nbl, tpb >>>(
            d_p.older_comm, d_p.node_comm, d_p.home_comm, V);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        { float t; cudaEventElapsedTime(&t, ev_start, ev_stop); ms_prepare += t; }

        // Phase 1: pick best community per node (warp-cooperative).
        // Block size 512 => 16 warps/block; each warp handles one node.
        // Per-warp shared memory is just cand_comm[PHASE1_WARP_CHUNK] ints;
        // per-candidate weight accumulators live in per-lane local memory
        // and are combined via deterministic warp-shuffle reductions.
        // Flavor branch: quality flavor uses the Gumbel-max probabilistic
        // variant which reads per-thread curand states.
        {
            int warps_per_block = tpb / 32;
            int warp_blocks = (V + warps_per_block - 1) / warps_per_block;
            size_t shared_bytes = (size_t)warps_per_block *
                (PHASE1_WARP_CHUNK * sizeof(int));

            cudaEventRecord(ev_start);
            if (flavor == 1) {
                find_community_phase1_warp_prob <<< warp_blocks, tpb, shared_bytes >>>(
                    d_p, d_g, d_rng_states, temperature);
            } else {
                find_community_phase1_warp <<< warp_blocks, tpb, shared_bytes >>>(d_p, d_g);
            }
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            { float t; cudaEventElapsedTime(&t, ev_start, ev_stop); ms_phase1 += t; }
        }

        // Phase 2 (Phase 2.5b deterministic variant): cross-node swap check
        // + community-size update.
        //
        // The original find_community_phase2 had two read-modify-write races:
        //   (a) reads d_p.final_comm[my_final] while other threads may be
        //       writing d_p.final_comm[my_final] (reverting their own move);
        //   (b) reads d_p.size[c] while other threads atomically update it.
        // Both made phase2's output non-deterministic on large graphs (e.g.
        // pcw13, merfish), which in turn produced run-to-run different labels
        // even after Phase 2.5 fixed phase1 and Phase 2.5b fixed update_partition.
        //
        // Fix: snapshot d_p.final_comm and d_p.size into read-only buffers
        // BEFORE the kernel runs; the new find_community_phase2_det kernel
        // reads exclusively from the snapshots (race-free) and writes each
        // node's decision into d_p.final_comm[i] (per-thread unique, race-free).
        // Then we rebuild d_p.size from scratch by scattering integer +1s
        // (integer atomicAdd is fully associative/commutative, deterministic).
        cudaEventRecord(ev_start);
        cudaMemcpyAsync(d_final_snapshot, d_p.final_comm,
                        V * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(d_size_snapshot,  d_p.size,
                        V * sizeof(int), cudaMemcpyDeviceToDevice);
        find_community_phase2_det <<< nbl, tpb >>>(
            d_p, d_g, d_final_snapshot, d_size_snapshot);
        // Rebuild size[] from the committed final_comm (deterministic).
        clear_size_kernel      <<< nbl, tpb >>>(d_p.size, V);
        accumulate_size_kernel <<< nbl, tpb >>>(d_p.final_comm, d_p.size, V);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        { float t; cudaEventElapsedTime(&t, ev_start, ev_stop); ms_phase2 += t; }

        // Phase 2.5b: deterministic move + community-stats recompute
        // (replaces the old atomicAdd-based update_partition).
        //
        //   1. apply_moves_assign_kernel: write d_p.node_comm[i] = d_p.final_comm[i]
        //   2. (implicit sync via separate kernel launch)
        //   3. apply_moves_home_kernel:   compute home_comm[i] from finalised node_comm
        //   4. recompute_community_stats_gpu: rebuild tot_in / tot_out / sum_in via
        //      thrust::sort_by_key + reduce_by_key + scatter (bit-deterministic).
        cudaEventRecord(ev_start);
        apply_moves_assign_kernel <<< nbl, tpb >>>(d_p, d_g);
        cudaDeviceSynchronize();
        apply_moves_home_kernel   <<< nbl, tpb >>>(d_p, d_g);
        cudaDeviceSynchronize();
        recompute_community_stats_gpu(
            d_p, d_g, V,
            d_sort_keys, d_sort_idx, d_sorted_vals,
            d_reduced_keys, d_reduced_vals,
            d_per_node_internal);
        cudaDeviceSynchronize();
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        { float t; cudaEventElapsedTime(&t, ev_start, ev_stop); ms_update += t; }

        // Count moves on GPU (only 4 bytes copied back)
        cudaEventRecord(ev_start);
        cudaMemset(d_move_count, 0, sizeof(int));
        count_moves_kernel<<< nbl, tpb >>>(d_p.node_comm, d_p.older_comm, d_move_count, V);
        cudaMemcpy(&moves, d_move_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        { float t; cudaEventElapsedTime(&t, ev_start, ev_stop); ms_count += t; }

        // Compute quality on GPU
        cudaEventRecord(ev_start);
        quality = find_quality_gpu(d_p, V, d_p.weight, d_p.resolution);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        { float t; cudaEventElapsedTime(&t, ev_start, ev_stop); ms_quality += t; }

        imp = quality - prev_quality;
        if (gpu_leiden_verbose) printf("new quality: %.6f  imp = %.6f\n", quality, imp);

    } while (moves > 0 && imp > 1e-5);

    if (gpu_leiden_verbose) {
        printf("PROFILE V=%d iters=%d  prepare=%.1f phase1=%.1f phase2=%.1f update=%.1f count=%.1f quality=%.1f total=%.1f ms\n",
               V, n_iters, ms_prepare, ms_phase1, ms_phase2, ms_update, ms_count, ms_quality,
               ms_prepare + ms_phase1 + ms_phase2 + ms_update + ms_count + ms_quality);
    }
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    cudaFree(d_move_count);

    // Copy data back from GPU ONCE after convergence (needed for aggregation phase)
    cudaMemcpy(p.node_comm, d_p.node_comm, V * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(p.sum_in, d_p.sum_in, V * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(p.tot_in, d_p.tot_in, V * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(p.tot_out, d_p.tot_out, V * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(p.older_comm, d_p.older_comm, V * sizeof(int), cudaMemcpyDeviceToHost);

    // Save the pre-refinement (local-moving) partition. Per Traag et al.
    // 2019, the Leiden output is the local-moving partition P, while the
    // refined partition P_ref is used only for aggregation. When this is
    // the terminal level (no further improvement expected), we must
    // compose tracked_labels with P, not P_ref, otherwise refinement
    // would shatter an already-optimal partition on the top level of a
    // second outer iteration.
    std::vector<int> pre_refine_comm(p.node_comm, p.node_comm + V);

    // Leiden refinement step (CPU). Replaces p.node_comm with the refined
    // partition and recomputes p.tot_in / p.tot_out / p.sum_in accordingly.
    // Refinement typically yields more (smaller) communities than the raw
    // local-moving result, which improves downstream ARI on large graphs.
    //
    // Phase 3.1 flavor branch: the quality flavor uses the probabilistic
    // refine variant, which samples from softmax(gain/temperature) over
    // positive-gain candidates. Seed mixes random_seed with V and a local
    // counter so different recursion depths and successive refine calls
    // within this Leiden_GPU invocation use distinct RNG streams.
    {
        auto refine_t0 = std::chrono::high_resolution_clock::now();
        if (flavor == 1) {
            refine_call_count++;
            unsigned int refine_seed =
                random_seed
                ^ (unsigned int)(0xDEADBEEFu + (unsigned int)V * 0x9E3779B9u)
                ^ (unsigned int)(refine_call_count * 0x85EBCA6Bu);
            refine_partition_cpu_prob(p, g, refine_seed, temperature);
        } else {
            refine_partition_cpu(p, g);
        }
        auto refine_t1 = std::chrono::high_resolution_clock::now();
        long refine_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             refine_t1 - refine_t0)
                             .count();
        if (gpu_leiden_verbose) printf("REFINE: %ld ms\n", refine_ms);
    }

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

    // Phase 2.5b working buffers
    cudaFree(d_sort_keys);
    cudaFree(d_sort_idx);
    cudaFree(d_sorted_vals);
    cudaFree(d_reduced_keys);
    cudaFree(d_reduced_vals);
    cudaFree(d_per_node_internal);
    cudaFree(d_final_snapshot);
    cudaFree(d_size_snapshot);
    // Phase 3.1 quality-flavor RNG states
    if (d_rng_states != nullptr) {
        cudaFree(d_rng_states);
        d_rng_states = nullptr;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
    if (gpu_leiden_verbose) {
        cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
        std::cout << "Leiden step completed in " << duration.count() << " minutes!" << std::endl;
        cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
        cout << "____________________________________________" << endl;
    }

    // Termination check: if refinement produced a partition in which
    // every node is in its own community (i.e., number of distinct
    // communities equals V), further aggregation would produce a graph
    // of the same size and we would be stuck in a fixed point. Count
    // the distinct community IDs in p.node_comm.
    int n_distinct_refined;
    {
        std::vector<int> tmp(p.node_comm, p.node_comm + V);
        std::sort(tmp.begin(), tmp.end());
        tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
        n_distinct_refined = (int)tmp.size();
    }

    // Phase 3.1: write this level's final modularity as a best-effort
    // value. The recursive Leiden_GPU call (if any) will overwrite it
    // with the deeper level's value. If the recursion bottoms out in
    // renumber_communities's early-return path (V<=1 or E==0), this
    // write ensures out_modularity still carries the latest level's
    // quality when the caller reads it. For non-recursive terminal
    // paths, this is redundant with the terminal-branch write below.
    if (out_modularity != NULL) {
        *out_modularity = quality;
    }

    if (quality > q_prev_it && n_distinct_refined < V)
    {
        // Non-terminal level: aggregate using the refined partition P_ref.
        // Compose tracked_labels with the REFINED p.node_comm, because the
        // next aggregation level creates one super-node per refined
        // sub-community. Subsequent local moving may re-merge sub-communities
        // that belong to the same pre-refinement parent community, matching
        // the standard Leiden paper algorithm.
        if (tracked_labels != NULL) {
            for (int i = 0; i < n_original; i++) {
                tracked_labels[i] = p.node_comm[tracked_labels[i]];
            }
        }
        renumber_communities(p, g, tracked_labels, n_original,
                             flavor, random_seed, temperature, out_modularity);
    }
    else
    {
        // Terminal level. Two cases:
        //   (a) local moving improved quality but refinement produced a
        //       trivial partition (n_distinct_refined == V). Use the
        //       refined partition — it's structurally the same as P.
        //   (b) local moving made no progress (quality <= q_prev_it).
        //       This is the "starting partition was already optimal" case;
        //       using the refined labels would shatter a good partition
        //       (happens on iter 2 when seeded from a good iter-1 result).
        //       Use the pre-refinement labels instead.
        // We pick between them based on whether local moving made progress.
        const int* final_labels_source =
            (quality > q_prev_it) ? p.node_comm : pre_refine_comm.data();
        if (tracked_labels != NULL) {
            for (int i = 0; i < n_original; i++) {
                tracked_labels[i] = final_labels_source[tracked_labels[i]];
            }
        }
        // Phase 3.1: write back the deepest-level modularity so the ILS
        // caller can pick the best restart. This path is the terminal
        // level of the recursion, so the value stored here corresponds
        // to the FINAL modularity of the original graph under the
        // hierarchical partition produced by this Leiden_GPU invocation.
        if (out_modularity != NULL) {
            *out_modularity = quality;
        }
        if (gpu_leiden_verbose) cout << "Leiden_GPU done and dusted :)" << endl;
    }

    return 0;
}

// ============================================================
// C API entry point for Python/external callers
// Accepts CSR arrays directly - no file I/O needed
// ============================================================

// Helper: (re)build the host-side graph struct from the caller's CSR arrays.
// Allocates fresh buffers and deep-copies; the caller is responsible for
// freeing any previously-allocated graph (via ::free(graph&)) before calling
// this. Used by leiden_from_csr both to set up the initial graph and to
// reload it between outer n_iterations passes, because Leiden_GPU mutates
// the graph in place during aggregation recursion.
static void build_graph_from_csr(graph& g,
                                 const int* out_indptr,
                                 const int* out_indices,
                                 const double* out_data,
                                 const int* in_indptr,
                                 const int* in_indices,
                                 const double* in_data,
                                 int n_nodes,
                                 int n_out_edges,
                                 int n_in_edges)
{
    g.nodes = n_nodes;
    // legacy field used by create_c_partition for nbrs/pos allocation
    g.ed = n_out_edges;

    g.out_col   = new int[n_nodes + 1];
    g.in_col    = new int[n_nodes + 1];
    g.child_out = new int[n_out_edges];
    g.child_in  = new int[n_in_edges];
    g.wts_out   = new double[n_out_edges];
    g.wts_in    = new double[n_in_edges];

    std::memcpy(g.out_col,   out_indptr,  (n_nodes + 1) * sizeof(int));
    std::memcpy(g.in_col,    in_indptr,   (n_nodes + 1) * sizeof(int));
    std::memcpy(g.child_out, out_indices, n_out_edges * sizeof(int));
    std::memcpy(g.child_in,  in_indices,  n_in_edges  * sizeof(int));
    std::memcpy(g.wts_out,   out_data,    n_out_edges * sizeof(double));
    std::memcpy(g.wts_in,    in_data,     n_in_edges  * sizeof(double));
}

// ============================================================================
// Phase 3.2: Shake perturbation for ILS "kick" moves.
//
// Gumbel-per-move noise (Phase 3.1) is too weak to escape strong local
// optima on large graphs — on pcw6 and merfish, the deterministic baseline
// already lands in a basin that probabilistic per-move noise cannot jump
// out of, so ILS early-stops without improvement.
//
// Textbook ILS instead uses coarse-grained "kick" perturbations: destroy a
// substantial chunk of the current solution, then re-run local search from
// there. For Leiden that translates to: pick a small number of large
// communities and shatter them back into singletons. The subsequent
// local-moving + refinement pass will rebuild them — and because the
// starting point is substantially different, it can land in a different
// (potentially better) basin.
//
// Implementation notes:
//   - Runs on the host over `labels[0..n)`, O(n + num_comm log num_comm).
//   - Label output must stay in [0, n_nodes) because create_c_partition_from_labels
//     uses labels as indices into partition arrays of length n_nodes.
//     We guarantee this via a two-phase scheme: mark shaken nodes with
//     unique negative sentinels, then run a compaction pass that collapses
//     both surviving labels and sentinels to [0, new_num_comm).
//   - Communities are sampled from the top-`top_frac` fraction by size
//     (default 20%) so we kick meaningful structure rather than wasting
//     a shake on a trivial community.
//   - k_shake distinct communities are picked uniformly from that top slice
//     via partial Fisher-Yates.
//
// Returns the number of nodes that were shaken (for logging / monitoring).
// ============================================================================
static int shake_partition(int* labels,
                           int n_nodes,
                           int k_shake,
                           double top_frac,
                           std::mt19937& rng,
                           bool verbose)
{
    if (n_nodes <= 1 || k_shake <= 0) return 0;

    // 1. Count community sizes.
    std::unordered_map<int, int> comm_size;
    comm_size.reserve(n_nodes / 4 + 1);
    for (int i = 0; i < n_nodes; i++) {
        comm_size[labels[i]]++;
    }
    int num_comm = (int)comm_size.size();
    if (num_comm < 2) return 0;

    // 2. Sort communities by size, descending.
    std::vector<std::pair<int, int>> sorted_comms;  // (size, comm_id)
    sorted_comms.reserve(num_comm);
    for (std::unordered_map<int, int>::iterator it = comm_size.begin();
         it != comm_size.end(); ++it) {
        sorted_comms.push_back(std::make_pair(it->second, it->first));
    }
    std::sort(sorted_comms.begin(), sorted_comms.end(),
              std::greater<std::pair<int, int> >());

    // 3. Determine the top-N slice.
    int top_n = (int)(num_comm * top_frac);
    if (top_n < 1) top_n = 1;
    if (top_n > num_comm) top_n = num_comm;

    // 4. Sample k_shake distinct indices from [0, top_n) via partial Fisher-Yates.
    int k = std::min(k_shake, top_n);
    std::vector<int> cand_idx(top_n);
    for (int i = 0; i < top_n; i++) cand_idx[i] = i;
    std::unordered_set<int> shake_set;  // community IDs to shatter
    shake_set.reserve(k);
    int shake_budget = 0;  // expected number of nodes to shake (for logging)
    for (int i = 0; i < k; i++) {
        std::uniform_int_distribution<int> dist(i, top_n - 1);
        int j = dist(rng);
        std::swap(cand_idx[i], cand_idx[j]);
        int picked_comm = sorted_comms[cand_idx[i]].second;
        shake_set.insert(picked_comm);
        shake_budget += sorted_comms[cand_idx[i]].first;
    }

    // 5. Mark shaken nodes with unique negative sentinels.
    int sentinel = -1;
    int shaken_count = 0;
    for (int i = 0; i < n_nodes; i++) {
        if (shake_set.find(labels[i]) != shake_set.end()) {
            labels[i] = sentinel--;
            shaken_count++;
        }
    }

    // 6. Compact labels to [0, new_num_comm). Every surviving label and every
    //    sentinel gets a fresh ID in iteration order, guaranteeing uniqueness
    //    for sentinels (= new singletons) and contiguity for the partition
    //    init routine that follows.
    std::unordered_map<int, int> relabel;
    relabel.reserve(num_comm - (int)shake_set.size() + shaken_count + 1);
    int next_new = 0;
    for (int i = 0; i < n_nodes; i++) {
        std::unordered_map<int, int>::iterator it = relabel.find(labels[i]);
        if (it == relabel.end()) {
            relabel[labels[i]] = next_new;
            labels[i] = next_new;
            next_new++;
        } else {
            labels[i] = it->second;
        }
    }

    if (verbose) {
        printf("shake: broke %d/%d top-%d communities, %d nodes -> singletons "
               "(num_comm %d -> %d)\n",
               (int)shake_set.size(), num_comm, top_n, shaken_count,
               num_comm, next_new);
    }
    return shaken_count;
}

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
    // Number of full Leiden passes. Each pass runs the entire local-moving
    // + refinement + aggregation hierarchy, starting from the previous
    // pass's final partition (first pass starts from singletons). A value
    // <= 0 means "use default" (2), matching leidenalg's default of
    // n_iterations=2. Quality is prioritised over speed; running 2 passes
    // typically improves ARI by 0.05-0.15 on real data.
    int max_iterations,
    unsigned int random_seed,
    // Output (caller-allocated)
    int* out_labels,            // [n_nodes] - filled with community ID per node
    // Phase 3.1 probabilistic / quality flavor parameters.
    int flavor,
    int n_restarts,
    double temperature,
    // Phase 3.3: verbose output flag. 0 = silence all the per-level
    // Leiden progress, kernel profile timings, ILS restart lines and
    // shake diagnostics. Non-zero restores the developer-facing output.
    // Python wrapper defaults to 0 so scanpy users get clean output.
    int verbose
)
{
    // Install the requested verbosity level globally for the duration
    // of this call. leiden_from_csr runs on a single host thread and
    // reads gpu_leiden_verbose from many helper functions inside the
    // shared library, so a global is the least invasive plumbing.
    gpu_leiden_verbose = verbose;

    // Resolve iteration count. Default to 2 passes (matching leidenalg).
    const int n_iters = (max_iterations <= 0) ? 2 : max_iterations;
    // Quality-flavor defaults. n_restarts < 0 means "use default" (4);
    // n_restarts == 0 is a valid "only run the mandatory deterministic
    // baseline, no probabilistic restarts" setting (useful for debug).
    // Default 4 matches the task spec; the mandatory det baseline adds
    // one more run for a total of 5 runs, which on large graphs still
    // stays within ~2x leidenalg's wall time thanks to the deterministic
    // path being 5-12x faster.
    const int n_runs  = (flavor == 1 && n_restarts < 0) ? 4 : n_restarts;
    // Temperature: Gumbel noise scale relative to 1/weight (the unit
    // of typical Leiden gain). 0.5 is the empirical sweet spot across
    // our benchmark graphs (smaller graphs respond better to this level
    // of perturbation; larger graphs are often near-optimal at det and
    // don't improve much regardless of T). 0 = deterministic argmax;
    // negative triggers the default.
    const double temp = (temperature < 0.0) ? 0.5 : temperature;

    if (flavor != 1) {
        // ================================================================
        // Deterministic path (unchanged from v0.2): single run with the
        // n_iterations outer loop from Phase 2.4d.
        // ================================================================
        int* current_labels = new int[n_nodes];
        for (int i = 0; i < n_nodes; i++) {
            current_labels[i] = i;
        }

        for (int iter = 0; iter < n_iters; iter++) {
            graph g;
            build_graph_from_csr(g,
                                 out_indptr, out_indices, out_data,
                                 in_indptr,  in_indices,  in_data,
                                 n_nodes, n_out_edges, n_in_edges);

            Leiden_Partition p;
            p.resolution = resolution;
            if (iter == 0) {
                create_c_partition(g, p);
            } else {
                create_c_partition_from_labels(g, p, current_labels);
            }

            if (gpu_leiden_verbose) {
                std::vector<int> tmp(p.node_comm, p.node_comm + n_nodes);
                std::sort(tmp.begin(), tmp.end());
                tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
                printf("leiden_from_csr: iter %d/%d starting with %zu distinct communities\n",
                       iter + 1, n_iters, tmp.size());
            }

            int* tracked_labels = new int[n_nodes];
            for (int i = 0; i < n_nodes; i++) {
                tracked_labels[i] = i;
            }

            // Deterministic flavor: pass flavor=0, any seed/temperature,
            // and nullptr for out_modularity (not used here).
            Leiden_GPU(p, g, n_out_edges, tracked_labels, n_nodes,
                       /*flavor=*/0, /*seed=*/random_seed,
                       /*temperature=*/0.0, /*out_modularity=*/nullptr);

            std::memcpy(current_labels, tracked_labels, n_nodes * sizeof(int));

            delete[] tracked_labels;
            free(g);
            free_part(p);
        }

        std::memcpy(out_labels, current_labels, n_nodes * sizeof(int));
        delete[] current_labels;
        return 0;
    }

    // ====================================================================
    // Quality path (flavor == 1): iterated local search over n_runs
    // seeds PLUS a mandatory deterministic baseline run. Each restart
    // runs the full n_iterations loop with a different base RNG seed.
    // We score each restart by computing the canonical UNWEIGHTED Q
    // directly from the tracked labels on the original graph (matching
    // the igraph / leidenalg reporting convention), and keep the
    // highest-Q result. Including the deterministic baseline guarantees
    // that quality flavor is never worse than deterministic flavor.
    // ====================================================================
    int*   best_labels     = new int[n_nodes];
    double best_modularity = -1e300;
    for (int i = 0; i < n_nodes; i++) best_labels[i] = i;

    // Precompute scalars used by the host-side Q evaluator.
    //
    // Phase 3.2 (FIX): Evaluate the WEIGHTED modularity — the same metric
    // leidenalg optimises and that the external benchmarks report via
    // `g.modularity(labels, weights='weight')`. Phase 3.1 evaluated the
    // UNWEIGHTED Q here, which turned out to let ILS "accept" restarts
    // whose weighted Q was actually worse than the deterministic baseline
    // on the merfish dataset — the unweighted-vs-weighted mismatch of the
    // scorer was silently corrupting our basin selection.
    //
    // Convention for the symmetric-CSR directed formulation (matches the
    // GPU kernel's modularity and igraph's undirected weighted Q):
    //   m_total  = sum of all out-edge weights (= 2W for symmetric input)
    //   k_out[v] = sum of out_data on v's out-edges (weighted out-degree)
    //   k_in[v]  = sum of in_data  on v's in-edges  (weighted in-degree)
    //   L_c (x2) = sum over (v in c, u in c, e = (v,u)) of out_data[e]
    //   K_c      = sum_{v in c} k_out[v] == sum_{v in c} k_in[v] for symmetric
    //   Q = (L_c_x2 - gamma * K_c * K_c / m_total) / m_total summed over c
    //
    // k_out / k_in depend only on the CSR data and are therefore
    // precomputed once before the restart loop. m_total ditto.
    double m_total = 0.0;
    for (int e = 0; e < n_out_edges; e++) m_total += out_data[e];
    const double inv_m = (m_total > 0.0) ? (1.0 / m_total) : 0.0;

    std::vector<double> k_out(n_nodes, 0.0);
    std::vector<double> k_in(n_nodes, 0.0);
    for (int v = 0; v < n_nodes; v++) {
        double s = 0.0;
        for (int e = out_indptr[v]; e < out_indptr[v + 1]; e++) s += out_data[e];
        k_out[v] = s;
    }
    for (int v = 0; v < n_nodes; v++) {
        double s = 0.0;
        for (int e = in_indptr[v]; e < in_indptr[v + 1]; e++) s += in_data[e];
        k_in[v] = s;
    }

    // Total number of "runs" = 1 mandatory deterministic baseline + n_runs
    // probabilistic restarts. The deterministic baseline gives us a
    // floor: the quality flavor's final output is guaranteed to be
    // at least as good as running deterministic flavor alone.
    //
    // Phase 3.2: we DROPPED the consecutive-non-improvement early stop
    // that Phase 3.1 used. With shake-based kick perturbation (below),
    // a run that fails to improve still leaves the search in a
    // meaningfully different region, so the next kick starts from a
    // different basin and can still break through. Early-stopping
    // wastes the exploration budget that made quality flavor quality.
    // Instead we track `no_improve` purely to adaptively *grow* the
    // shake intensity when we plateau.
    const int total_runs = n_runs + 1;
    int no_improve = 0;

    // RNG for shake perturbation (independent of the Gumbel RNG on the
    // GPU so their seed streams don't interact). Seeded off random_seed
    // with a large xor constant for decorrelation.
    std::mt19937 shake_rng(random_seed ^ 0xDEADBEEFu);

    for (int restart = 0; restart < total_runs; restart++) {
        // Phase 3.2c: ALL restarts now use the deterministic local search
        // path (flavor=0). Diversity comes entirely from the shake
        // perturbation applied to the warm-start partition — layering
        // per-move Gumbel noise on top of an already-perturbed starting
        // point turned out to prevent the local search from climbing
        // cleanly to the shaken basin's optimum. Classical ILS recipe:
        //   S' = perturbation(S)    (shake)
        //   S'' = local_search(S')  (pure greedy, NOT stochastic)
        //   accept if better(S'', S*)
        // Restart 0 = baseline (no shake); restarts 1..N = kick + refine.
        int run_flavor = 0;
        // Decorrelate restart seeds with a large odd prime (golden ratio
        // hash constant). With flavor=0 the seed is unused by Leiden_GPU
        // itself, but we still reseed the shake RNG per restart (below)
        // via `this_seed` indirectly through the shake call sequence.
        unsigned int this_seed =
            random_seed + (unsigned int)(restart * 2654435761u);

        // current_labels across the n_iters outer loop for this restart.
        // Phase 3.2: probabilistic restarts WARM-START from the current
        // best partition and then SHAKE (shatter k large communities
        // into singletons). This implements the classical ILS "kick":
        // destroy a substantial chunk of the current solution, then
        // re-run local search from there. Without a kick, probabilistic
        // Leiden restarts converge back to the same basin — which is
        // exactly what we saw on pcw6 and merfish under Phase 3.1.
        int* current_labels = new int[n_nodes];
        int shaken_nodes = 0;
        if (restart == 0) {
            // Deterministic baseline: pure singletons (unchanged from v0.2).
            for (int i = 0; i < n_nodes; i++) current_labels[i] = i;
        } else {
            // Warm-start from best so far.
            std::memcpy(current_labels, best_labels, n_nodes * sizeof(int));
            // Adaptive shake intensity: start gentle (k=1), grow with
            // consecutive non-improving restarts so we kick harder when
            // the current basin is sticky. Caps implicitly at top_n
            // inside shake_partition.
            int k_shake = 1 + (no_improve / 2);  // 1,1,2,2,3,3,...
            shaken_nodes = shake_partition(
                current_labels, n_nodes,
                /*k_shake=*/k_shake,
                /*top_frac=*/0.20,
                shake_rng,
                /*verbose=*/(bool)gpu_leiden_verbose);
        }

        for (int iter = 0; iter < n_iters; iter++) {
            graph g;
            build_graph_from_csr(g,
                                 out_indptr, out_indices, out_data,
                                 in_indptr,  in_indices,  in_data,
                                 n_nodes, n_out_edges, n_in_edges);

            Leiden_Partition p;
            p.resolution = resolution;
            // Phase 3.2: at iter 0 of a probabilistic restart, current_labels
            // already holds the shaken warm-start partition, so feed it
            // directly to the C partition builder. For the deterministic
            // baseline (restart 0) we keep the pure singleton path.
            if (restart == 0 && iter == 0) {
                create_c_partition(g, p);
            } else {
                create_c_partition_from_labels(g, p, current_labels);
            }

            if (gpu_leiden_verbose) {
                std::vector<int> tmp(p.node_comm, p.node_comm + n_nodes);
                std::sort(tmp.begin(), tmp.end());
                tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
                const char* kind = (restart == 0) ? "DET" : "KICK";
                printf("leiden_from_csr: %s run %d/%d iter %d/%d seed=%u starting with %zu distinct communities\n",
                       kind, restart, n_runs, iter + 1, n_iters, this_seed, tmp.size());
            }

            int* tracked_labels = new int[n_nodes];
            for (int i = 0; i < n_nodes; i++) tracked_labels[i] = i;

            double final_modularity = -1e300;
            unsigned int iter_seed =
                this_seed ^ (unsigned int)(iter * 0x9E3779B9u + 0x13579BDFu);
            Leiden_GPU(p, g, n_out_edges, tracked_labels, n_nodes,
                       /*flavor=*/run_flavor, iter_seed, temp, &final_modularity);

            std::memcpy(current_labels, tracked_labels, n_nodes * sizeof(int));

            delete[] tracked_labels;
            free(g);
            free_part(p);
        }

        // Phase 3.2: Compute WEIGHTED modularity from current_labels
        // on the original CSR structure. This matches leidenalg's
        // internal optimisation target AND the external benchmarks'
        // reporting convention (`g.modularity(labels, weights='weight')`).
        //
        //   Q = (1/m_total) * sum_c { internal_w(c) - res * K_c_out * K_c_in / m_total }
        //
        // where internal_w(c) = sum over directed arcs (v,u) with both
        //                       endpoints in c of out_data[(v,u)],
        //       K_c_out       = sum_{v in c} weighted out-degree,
        //       K_c_in        = sum_{v in c} weighted in-degree.
        //
        // For a symmetric undirected CSR each undirected edge contributes
        // its weight twice to internal_w (once in each direction), and
        // m_total == 2W. The formula simplifies to the standard
        // undirected weighted modularity after that factor cancels.
        double run_modularity = -1e300;
        if (m_total > 0.0) {
            // Community totals (weighted)
            std::unordered_map<int, double> tot_out_c;
            std::unordered_map<int, double> tot_in_c;
            std::unordered_map<int, double> internal_c;
            for (int v = 0; v < n_nodes; v++) {
                int c = current_labels[v];
                tot_out_c[c] += k_out[v];
                tot_in_c[c]  += k_in[v];
            }
            // Internal edge weight sum: iterate all directed arcs via out-CSR
            for (int v = 0; v < n_nodes; v++) {
                int cv = current_labels[v];
                for (int e = out_indptr[v]; e < out_indptr[v + 1]; e++) {
                    int u = out_indices[e];
                    if (current_labels[u] == cv) {
                        internal_c[cv] += out_data[e];  // WEIGHTED
                    }
                }
            }
            double sum_internal = 0.0, sum_expected = 0.0;
            for (std::unordered_map<int, double>::iterator it = tot_out_c.begin();
                 it != tot_out_c.end(); ++it) {
                int c = it->first;
                sum_internal += internal_c[c];
                sum_expected += resolution * tot_out_c[c] * tot_in_c[c] * inv_m;
            }
            run_modularity = (sum_internal - sum_expected) * inv_m;
        }

        // Require a meaningful improvement to accept a probabilistic
        // restart over the current best. This is important because
        // tied-Q partitions can differ substantially in their ARI
        // against a reference (e.g. leidenalg with a fixed seed): a
        // noise-driven walk can reach a slightly higher unweighted Q
        // but a more different label structure. Demanding a threshold
        // improvement protects against the "worse ARI, marginally
        // higher Q" failure mode while still picking up real
        // improvements from probabilistic exploration.
        //
        // Phase 3.2: lowered from 1e-3 to 1e-4. Shake perturbation gives
        // the search genuine basin-hopping capability, so tighter
        // improvements are now meaningful signals rather than noise.
        //
        // The deterministic baseline (first run) always unconditionally
        // sets the initial best — it sees best_modularity == -1e300.
        const double improve_threshold = 1e-4;
        bool improved;
        if (run_flavor == 0) {
            improved = (run_modularity > best_modularity);
        } else {
            improved = (run_modularity > best_modularity + improve_threshold);
        }

        if (improved) {
            best_modularity = run_modularity;
            std::memcpy(best_labels, current_labels, n_nodes * sizeof(int));
            // Reset shake intensity when we land a real improvement:
            // the new basin is fresh, start kicking gently again.
            no_improve = 0;
        } else if (restart > 0) {
            // Grow shake intensity for the next kick restart.
            no_improve++;
        }

        if (gpu_leiden_verbose) {
            const char* kind_final = (restart == 0) ? "DET  " : "KICK ";
            if (restart > 0 && shaken_nodes > 0) {
                printf("ILS %s run %d/%d: modularity = %.6f (best so far: %.6f)%s  "
                       "[shaken=%d, k=%d]\n",
                       kind_final, restart, n_runs, run_modularity, best_modularity,
                       improved ? " [accepted]" : "",
                       shaken_nodes, 1 + ((no_improve - (improved ? 0 : 1)) / 2));
            } else {
                printf("ILS %s run %d/%d: modularity = %.6f (best so far: %.6f)%s\n",
                       kind_final, restart, n_runs, run_modularity, best_modularity,
                       improved ? " [accepted]" : "");
            }
        }

        delete[] current_labels;

        // Phase 3.2: no more early-stop. Full restart budget always runs
        // so that progressively harder kicks get a chance to break
        // through on graphs where the deterministic baseline is sticky.
    }

    std::memcpy(out_labels, best_labels, n_nodes * sizeof(int));
    delete[] best_labels;
    return 0;
}

}  // extern "C"
