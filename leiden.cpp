#include <iostream>
#include "struct.h"
#include <algorithm>
#include "leiden.h"
#include <stdio.h>
#include <vector>
#include <iomanip>
#include <random>
#include <omp.h>

using namespace std;


int Leiden_CPU(Leiden_Partition& p, graph& gr)
{
    double quality = 0.0;
    double newGain = 0.0;
    int best_comm, old_comm, mvs;
    double imp = 0.0;
    double prev_quality = 0.0;
    double q_prev_it = 0;
    double edg_wt = p.weight;

    quality = cal_quality(p.sum_in, p.tot_in, p.tot_out, gr.nodes, p.weight, p.resolution);
    cout << "old quality: " << quality << endl;
    q_prev_it = quality;

    time_t start, end;
    time(&start);
    int counter = 0;

    double dncomm;

    do
    {
        counter++;
        prev_quality = quality;
        mvs = 0;

        for (int i = 0; i < gr.nodes; i++)
        {
            double bestGain = 0.0;
            double old_gain = 0.0;
            old_comm = p.node_comm[i];
            int comm = 0;

   
            double dnc = 0.0;
            for (int neighbour = gr.out_col[i]; neighbour < gr.out_col[i+1]; neighbour++)
            {
                if (gr.child_out[neighbour] != i && p.node_comm[gr.child_out[neighbour]] == p.node_comm[i])
                {
                    dnc += gr.wts_out[neighbour];
                }
            }

            for (int neighbour = gr.in_col[i]; neighbour < gr.in_col[i+1]; neighbour++)
            {
                if (gr.child_in[neighbour] != i && p.node_comm[gr.child_in[neighbour]] == p.node_comm[i])
                {
                    dnc += gr.wts_in[neighbour];
                }
            }

            p.sum_in[old_comm] -= dnc + p.self_loops[i];
            p.tot_in[old_comm] -= p.in_deg[i];
            p.tot_out[old_comm] -= p.out_deg[i];


            for (int community = p.pos[i]; community < p.pos[i+1]; community++)
            {
                int comm = p.node_comm[p.nbrs[community]];
                double dncomm = 0.0;

                for (int neighbour = gr.out_col[i]; neighbour < gr.out_col[i+1]; neighbour++)
                {
                    if (i != gr.child_out[neighbour] && p.node_comm[gr.child_out[neighbour]] == comm)
                    {
                        dncomm += gr.wts_out[neighbour];
                    }
                }

                for (int neighbour = gr.in_col[i]; neighbour < gr.in_col[i+1]; neighbour++)
                {
                    if (i != gr.child_in[neighbour] && p.node_comm[gr.child_in[neighbour]] == comm)
                    {
                        dncomm += gr.wts_in[neighbour];
                    }
                }

                newGain = (dncomm + p.self_loops[i]) / p.weight -
                          p.resolution * ((p.tot_in[comm] * p.out_deg[i] + p.tot_out[comm] * p.in_deg[i]) /
                           (p.weight * p.weight));

                if (newGain > bestGain)
                {
                    bestGain = newGain;
                    best_comm = comm;
                    dnc = dncomm;
                }
            }

            // Move node to best community
            p.node_comm[i] = best_comm;
            p.sum_in[best_comm] += dnc + p.self_loops[i];
            p.tot_in[best_comm] += p.in_deg[i];
            p.tot_out[best_comm] += p.out_deg[i];

            if (best_comm != old_comm)
            {
                mvs++;
            }
        }

        quality = cal_quality(p.sum_in, p.tot_in, p.tot_out, gr.nodes, p.weight, p.resolution);
        imp = quality - prev_quality;
        cout << "new quality: " << quality << "  imp  = " << imp << endl;

    } while (mvs > 0 && imp > 0.005);
 time(&end); 
 double time_taken = double(end - start) * 1000;
 cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
 cout << "Leiden step completed in " << fixed  << time_taken << setprecision(6); 
 cout << " sec " << endl; 
 cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;

cout << endl;
if(quality>q_prev_it)
{
  c_renumber_communities(p,gr);
}
else
{
  cout << "Leiden done and dusted :)" << endl;
}
  return 0;
}


double cal_quality(double in[], double tot_in[], double tot_out[],long int size, double edgs, double resolution)
{
   double q=0;
   for(long int i=0; i< size; i++)
   {
      if (tot_in[i] > 0 || tot_out[i] > 0)
      {
        q+=in[i]-resolution*(tot_in[i]*tot_out[i]/(edgs));
      }
   }
          q= q/(edgs);

     return q;
}

graph next_pahse(aggregate_adj& ad, graph& g)
 {
  g.child_out =  new int [ad.edges];
  g.child_in =  new int [ad.edges];
  g.wts_out = new double [ad.edges];
  g.wts_in = new double [ad.edges];


int weight_index =0;
int node_index = 0;
for(int i=0; i < ad.next_graph.size(); i++)
{ 
 for(int j=0; j< ad.next_graph[i].size(); j++)
  {
    g.child_out[node_index++]=ad.next_graph[i][j].second;
    g.wts_out[weight_index++]=ad.next_graph[i][j].first;
  }
 }
int weight_index1 =0;
int node_index1 = 0;
for(int i=0; i < ad.in_neighbours.size(); i++)
{ 
  const vector <pair <double,int > > neighs= ad.in_neighbours[i];
 for(int j=0; j<neighs.size(); j++)
 {
	 g.child_in[node_index1++]= neighs[j].second;
	 g.wts_in[weight_index1++]= neighs[j].first;
 }
}
  return g; 
}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
int c_renumber_communities(Leiden_Partition& p, graph& g)
{
unordered_map <int, vector <int> >comms;  

for(int comm=0; comm< g.nodes; comm++)
{
    comms[p.node_comm[comm]].push_back(comm);
   
}

vector<int> renumber_c;     
int community_range=0;
for (auto it = comms.begin(); it != comms.end(); ++it)
{
    renumber_c.push_back(it->first);
    community_range++;
}
 cout << "range is: " << community_range << endl << endl;


sort(renumber_c.begin(), renumber_c.end());
unordered_map < int, int> dummy_indexes;                   
for (int it = 0; it<renumber_c.size(); it++)
{
  dummy_indexes[renumber_c[it]]=it;
}
 auto start_time = std::chrono::high_resolution_clock::now();
 aggregate_adj adj;

// Pre-size the output so threads can write to fixed indices instead of push_back
adj.next_graph.resize(community_range);

#pragma omp parallel for
for(int comm=0; comm<community_range; comm++)
{
int community=renumber_c[comm];
const vector<int> & community_nodes = comms[community];
unordered_map <int, double> temp;
vector <pair <double, int> > temp_edges;
for(int node=0; node<community_nodes.size(); node++)
{
for(int neighbor=g.out_col[community_nodes[node]]; neighbor<g.out_col[community_nodes[node]+1]; neighbor++)
{
    int n_neig=g.child_out[neighbor];
    double w_neig=g.wts_out[neighbor];
    int neig_comm= p.node_comm[n_neig];
    temp[dummy_indexes[neig_comm]]+= w_neig;
}
}

for (const auto& entry : temp)
{
temp_edges.push_back(make_pair(entry.second, entry.first));
}
    auto myCompare= [](const pair<double, int>& a, const pair<double, int>& b) {
        return a.second < b.second;
    };
sort(temp_edges.begin(), temp_edges.end(), myCompare);
adj.next_graph[comm] = std::move(temp_edges);
}

int arcs=0;
for(int i=0; i< adj.next_graph.size(); i++)
{
for(int j=0; j< adj.next_graph[i].size(); j++)
{
  adj.in_neighbours[adj.next_graph[i][j].second].push_back({adj.next_graph[i][j].first, i});
}
  arcs+=adj.next_graph[i].size();
}

g.nodes=adj.next_graph.size();
adj.len= adj.next_graph.size();
adj.edges=arcs;
g.out_col = new int [(adj.len+1)];
g.in_col =  new int [(adj.len+1)];
g.out_col[0]=0; 


int valu=0;
for (int idx=0; idx<adj.next_graph.size(); idx++) 
{
  valu+= adj.next_graph[idx].size();
  g.out_col[(idx+1)]=valu;
}
g.in_col[0]=0; 
int s=0;
for (int idx=0; idx<adj.in_neighbours.size(); idx++) 
{
  s+=adj.in_neighbours[idx].size();
  g.in_col[idx+1]=s;
}
  g.child_out =  new int [adj.edges];
  g.child_in =  new int [adj.edges];
  g.wts_out = new double [adj.edges];
  g.wts_in = new double [adj.edges];

int weight_index =0;
int node_index = 0;
for(int i=0; i < adj.next_graph.size(); i++)
{ 
 for(int j=0; j< adj.next_graph[i].size(); j++)
  {
    g.child_out[node_index++]=adj.next_graph[i][j].second;
    g.wts_out[weight_index++]=adj.next_graph[i][j].first;
  }
 }
int weight_index1 =0;
int node_index1 = 0;
for(int i=0; i < adj.in_neighbours.size(); i++)
{ 
  const vector <pair <double,int > > neighs= adj.in_neighbours[i];
 for(int j=0; j<neighs.size(); j++)
 {
	 g.child_in[node_index1++]= neighs[j].second;
	 g.wts_in[weight_index1++]= neighs[j].first;
 }
}
auto end_time = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
std::cout << "Aggreagte step on Host completed in " << duration.count() << " minutes!" << std::endl; 
cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
cout << "____________________________________________" << endl;
create_c_partition(g,p);
Leiden_CPU(p, g);
return 0;
}                                                                            
 /*`````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````*/
                                                                                                    
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
Leiden_Partition create_c_partition(graph& g, Leiden_Partition& p)
{  
  p.in_deg = new double [g.nodes];
  p.out_deg = new double [g.nodes];
  p.tot_in = new double [g.nodes];
  p.tot_out = new double [g.nodes];
  p.sum_in = new double [g.nodes];
  p.sum_kin = new double [g.nodes];
  p.self_loops = new double [g.nodes];
  p.node_comm = new int [g.nodes];
  p.size = new int [g.nodes];
  p.home_comm = new double [g.nodes];
  p.final_comm=new int [g.nodes];
  p.nbrs=new int [g.nodes + g.ed];
  p.older_comm= new int[g.nodes];
  p.pos=new int [g.nodes + 1];


for (int i=0; i< (g.nodes+g.ed); i++)
{ 
  p.nbrs[i]=0;
}

for (int i=0; i< (g.nodes+1); i++)
{ 
  p.pos[i]=0;
}

for (int i=0; i< g.nodes; i++)
{ 
  p.node_comm[i]=i;
  p.size[i]=1;
  p.final_comm[i]=i;
  p.in_deg[i]=0;
  p.home_comm[i]=0;
  p.out_deg[i]=0;
  p.tot_in[i]=0;
  p.tot_out[i]=0;
  p.sum_kin[i]=0;
  p.self_loops[i]=0;
  p.sum_in[i]=0;
  p.older_comm[i]=i;
  p.weight=0;
}
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
for (int i=0; i<g.nodes; i++)
{ 
  p.in_deg[i]=indegree(g, p, i);
  p.out_deg[i]=outdegree(g, p, i);
  p.weight=p.weight+p.out_deg[i];
  p.tot_in[i]=p.in_deg[i];
  p.tot_out[i]=p.out_deg[i];
  p.self_loops[i]=selfloop(g, p, i);
  p.sum_in[i]=p.self_loops[i];
}

p.count.clear();
p.neigh_commNb.clear();
p.neigh_pos.clear();

int tot_inc=0;
p.count.push_back(0);
for (int j=0; j<g.nodes; j++)
{
  int inc=0;
  p.neigh_commNb.push_back(j);
for (int i=g.out_col[j]; i<g.out_col[j+1]; i++)
{
  if(j!=p.node_comm[g.child_out[i]])
  {
  p.neigh_commNb.push_back(p.node_comm[g.child_out[i]]);
  inc++;
  }
}
tot_inc+=inc;
p.count.push_back(tot_inc);
}

for (int i=0; i< p.neigh_commNb.size(); i++)
{
  p.nbrs[i]=p.neigh_commNb[i];
}

p.neigh_pos.push_back(0); 
int pos=0;
for (int j=1; j<g.nodes+1; j++)
{
  int b=p.count[j]-p.count[j-1];
  int index=p.neigh_pos[j-1];
  p.neigh_pos.push_back(index+b+1);
}

for (int i=0; i< p.neigh_pos.size(); i++)
{
  p.pos[i]=p.neigh_pos[i];
}

    return p;
}

inline double selfloop(graph& g, Leiden_Partition& p, int v)
{
 for(int j=g.out_col[v]; j< g.out_col[v+1]; j++)
  {  int node=v;
  if(node==g.child_out[j])
  {
    p.self_loops[v]+=g.wts_out[j];
  }
 }  
 return p.self_loops[v];
}


inline double indegree(graph& g, Leiden_Partition& p, int v)
{
  for(int j=g.in_col[v]; j< g.in_col[v+1]; j++)
  {
   p.in_deg[v]+=g.wts_in[j];
  }  
  return p.in_deg[v];
}

inline double outdegree(graph& g, Leiden_Partition& p, int v)
 {
  for(int j=g.out_col[v]; j< g.out_col[v+1]; j++)
  {
   p.out_deg[v]+=g.wts_out[j];
  }  
  return p.out_deg[v];
 }

graph graph_process (adjlist& adj, graph& g)
{
  g.out_col =  new int [(adj.len+1)];
  g.in_col =  new int [(adj.len+1)];
  g.child_out =  new int [adj.edges];
  g.child_in =  new int [adj.edges];
  g.wts_out = new double [adj.edges];
  g.wts_in = new double [adj.edges];


 g.out_col[0]=0; 

int sum=0;
for (int idx=0; idx<adj.out_gr.size(); idx++) 
{
  sum+=adj.out_gr[idx].size();
  g.out_col[idx+1]=sum;
}

g.in_col[0]=0;

int value=0;
for (int idx=0; idx<adj.in_gr.size(); idx++) 
{
  value+=adj.in_gr[idx].size();
  g.in_col[idx+1]=value;

}


int weight_index =0;
int node_index = 0;
for(int i=0; i < adj.out_gr.size(); i++)
{ 
 for(int j=0; j< adj.out_gr[i].size(); j++)
  {
   g.child_out[node_index++]=adj.out_gr[i][j];
   g.wts_out[weight_index++]=adj.out_wt[i][j];
 
  }
 }

int weight_index1 =0;
int node_index1 = 0;
for(int i=0; i < adj.in_gr.size(); i++)
{ 
 for(int j=0; j< adj.in_gr[i].size(); j++)
 {
	 g.child_in[node_index1++]= adj.in_gr[i][j];
	 g.wts_in[weight_index1++]= adj.in_wt[i][j];
 }
}
  return g; 
}
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void free(graph& g)
{
  delete[] g.child_in;
  delete[] g.child_out;
  delete[] g.wts_out;
  delete[] g.wts_in;
  delete[] g.in_col;
  delete[] g.out_col;
}
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
void free_part(Leiden_Partition& p)
{
  delete[] p.tot_out;
  delete[] p.tot_in;
  delete[] p.sum_in;
  delete[] p.in_deg;
  delete[] p.out_deg;
  delete[] p.node_comm;
  delete[] p.home_comm;
  delete[] p.final_comm;
  delete[] p.older_comm;
  delete[] p.sum_kin;
  delete[] p.self_loops;
  delete[] p.nbrs;
  delete[] p.pos;
  delete[] p.size;
}
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
// Leiden refinement (CPU, OpenMP over parent communities).
//
// Replaces p.node_comm (currently the local-moving partition P) with the
// refined partition P_refined. Each parent community c in P is refined
// independently by starting its members as singletons and running a
// constrained greedy local move (candidate sub-communities restricted to
// nodes whose parent community is c).
//
// After refinement, p.tot_in, p.tot_out, and p.sum_in are recomputed for
// the refined partition using the same convention as Leiden_CPU:
//   sum_in[c] counts self-loops once per node, and for each (u,v) internal
//   edge with both endpoints in c, counts it once from out-edges and once
//   from in-edges (i.e., the internal weight is effectively doubled for
//   directed-pair edges, matching the existing code).
void refine_partition_cpu(Leiden_Partition& p, graph& g)
{
    int V = g.nodes;
    double w = p.weight;
    double res = p.resolution;

    // 1) save parent partition P
    std::vector<int> orig_comm(V);
    for (int i = 0; i < V; i++) orig_comm[i] = p.node_comm[i];

    // 2) reset to singletons
    for (int i = 0; i < V; i++) p.node_comm[i] = i;

    // Group vertices by parent community. We use a dense bucket based on
    // the range of orig_comm values so iteration order is deterministic and
    // we can trivially parallelize over distinct parent communities.
    // Use std::map to get deterministic ordered iteration (important for
    // reproducibility across runs).
    std::map<int, std::vector<int> > by_orig;
    for (int i = 0; i < V; i++) {
        by_orig[orig_comm[i]].push_back(i);
    }

    std::vector<int> orig_ids;
    orig_ids.reserve(by_orig.size());
    for (std::map<int, std::vector<int> >::iterator it = by_orig.begin();
         it != by_orig.end(); ++it) {
        orig_ids.push_back(it->first);
    }

    // 3) refine each parent community independently
    int n_parents = (int)orig_ids.size();
    #pragma omp parallel for schedule(dynamic, 1)
    for (int idx = 0; idx < n_parents; idx++) {
        int c = orig_ids[idx];
        const std::vector<int>& verts = by_orig[c];
        if (verts.size() <= 1) continue;  // singleton parent — nothing to refine

        // Local sub-community degree stats, keyed by sub-community id (node id).
        // std::map for deterministic iteration order.
        std::map<int, double> sub_tot_in;
        std::map<int, double> sub_tot_out;
        std::map<int, double> sub_sum_in;

        for (size_t k = 0; k < verts.size(); k++) {
            int v = verts[k];
            sub_tot_in[v]  = p.in_deg[v];
            sub_tot_out[v] = p.out_deg[v];
            sub_sum_in[v]  = p.self_loops[v];
        }

        double inv_w  = 1.0 / w;
        double inv_w2 = inv_w * inv_w;

        // Greedy sweep: each vertex tries to join the best sub-community
        // composed of neighbors whose PARENT community is still c.
        for (size_t k = 0; k < verts.size(); k++) {
            int v = verts[k];
            int my_sub = p.node_comm[v];

            // Accumulate weight from v to each candidate sub (use std::map
            // for deterministic iteration order).
            std::map<int, double> dnc;
            for (int e = g.out_col[v]; e < g.out_col[v + 1]; e++) {
                int u = g.child_out[e];
                if (u == v) continue;
                if (orig_comm[u] != c) continue;
                dnc[p.node_comm[u]] += g.wts_out[e];
            }
            for (int e = g.in_col[v]; e < g.in_col[v + 1]; e++) {
                int u = g.child_in[e];
                if (u == v) continue;
                if (orig_comm[u] != c) continue;
                dnc[p.node_comm[u]] += g.wts_in[e];
            }

            double dv_in  = p.in_deg[v];
            double dv_out = p.out_deg[v];
            double sl_v   = p.self_loops[v];

            // Remove v from its current sub (bookkeeping mirrors Leiden_CPU).
            double dnc_self = 0.0;
            std::map<int, double>::iterator it_self = dnc.find(my_sub);
            if (it_self != dnc.end()) dnc_self = it_self->second;

            sub_tot_in[my_sub]  -= dv_in;
            sub_tot_out[my_sub] -= dv_out;
            sub_sum_in[my_sub]  -= dnc_self + sl_v;

            // Find best candidate sub-community.
            int best_sub = my_sub;
            double best_gain = 0.0;
            for (std::map<int, double>::iterator it = dnc.begin();
                 it != dnc.end(); ++it) {
                int sub = it->first;
                double dncomm = it->second;
                double toc_in  = sub_tot_in[sub];
                double toc_out = sub_tot_out[sub];
                double gain = (dncomm + sl_v) * inv_w
                            - res * (toc_in * dv_out + toc_out * dv_in) * inv_w2;
                if (gain > best_gain) {
                    best_gain = gain;
                    best_sub = sub;
                }
            }

            // Re-insert v into best_sub (may equal my_sub).
            double best_dnc = 0.0;
            std::map<int, double>::iterator it_best = dnc.find(best_sub);
            if (it_best != dnc.end()) best_dnc = it_best->second;

            sub_tot_in[best_sub]  += dv_in;
            sub_tot_out[best_sub] += dv_out;
            sub_sum_in[best_sub]  += best_dnc + sl_v;

            p.node_comm[v] = best_sub;
        }
    }

    // 4) Recompute global tot_in / tot_out / sum_in for the refined partition.
    for (int i = 0; i < V; i++) {
        p.tot_in[i]  = 0.0;
        p.tot_out[i] = 0.0;
        p.sum_in[i]  = 0.0;
    }
    for (int v = 0; v < V; v++) {
        int c = p.node_comm[v];
        p.tot_in[c]  += p.in_deg[v];
        p.tot_out[c] += p.out_deg[v];
    }
    for (int v = 0; v < V; v++) {
        int cv = p.node_comm[v];
        // Self-loops counted once (matches convention: dnc excludes self-loops,
        // self_loops[v] is added separately in Leiden_CPU bookkeeping).
        p.sum_in[cv] += p.self_loops[v];
        for (int e = g.out_col[v]; e < g.out_col[v + 1]; e++) {
            int u = g.child_out[e];
            if (u == v) continue;
            if (p.node_comm[u] == cv) {
                p.sum_in[cv] += g.wts_out[e];
            }
        }
        for (int e = g.in_col[v]; e < g.in_col[v + 1]; e++) {
            int u = g.child_in[e];
            if (u == v) continue;
            if (p.node_comm[u] == cv) {
                p.sum_in[cv] += g.wts_in[e];
            }
        }
    }
}

