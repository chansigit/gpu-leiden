#include "struct.h"
#include <thread>
#include <set>
#include <algorithm>
#include "leiden.h"


using namespace std;


int main(int argc, char* argv[])
{

  if (argc < 3) {
        cout << "Usage: " << argv[0] << " <input_file> <cpu|gpu> [resolution]" << endl;
        return 1;
    }

string filename = argv[1];
string mode = argv[2];
double resolution = 1.0;
if (argc >= 4) {
    resolution = atof(argv[3]);
}
cout << "Resolution: " << resolution << endl;
std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
ifstream file(filename);
auto start_time = std::chrono::high_resolution_clock::now();
    if (!file.is_open()) {
        cout << "Failed to open the file: " << filename << endl;
        return 1;
    }

adjlist adj;
    adj.edges = 0;

    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        int v1, v2;
        float w;
while (iss >> v1 >> v2 >> w) {
            adj.out_gr[v1].push_back(v2);
            adj.out_wt[v1].push_back(w);
            adj.in_gr[v2].push_back(v1);
            adj.in_wt[v2].push_back(w);
            adj.edges++;
        }
}
    file.close();


adj.len=adj.out_gr.size();


 auto end_time = std::chrono::high_resolution_clock::now();
 auto duration = std::chrono::duration_cast<std::chrono::minutes>(end_time - start_time);
 std::cout << "Time taken for reading first graph is " << duration.count() << " minutes!" << std::endl;

 Leiden_Partition part;
 part.resolution = resolution;
 graph g;
 g.nodes= adj.len;
 graph_process(adj, g);
 g.ed=adj.edges;
 create_c_partition(g,part);
 int arr_size= adj.edges;

if (mode == "cpu") {
    Leiden_CPU(part, g);
} else if (mode == "gpu") {
    // Phase 3.1: default to deterministic flavor for the legacy CLI test,
    // which is what tests/verify.sh captures as its GPU baseline.
    Leiden_GPU(part, g, arr_size, NULL, 0,
               /*flavor=*/0, /*seed=*/0, /*temperature=*/0.0,
               /*out_modularity=*/NULL);

} else if (mode == "gpu_csr") {
    // Test the leiden_from_csr C API using the existing graph
    // (exercises the new API end-to-end with known test data)
    int* labels = new int[g.nodes];
    leiden_from_csr(
        g.out_col, g.child_out, g.wts_out, g.ed,
        g.in_col,  g.child_in,  g.wts_in,  g.ed,
        g.nodes,
        resolution,
        -1,  // max_iterations
        0,   // random_seed
        labels,
        /*flavor=*/0,
        /*n_restarts=*/0,
        /*temperature=*/0.0,
        /*verbose=*/1  // CLI mode: keep full output for debugging
    );
    // Print a summary: number of unique labels (= number of final communities)
    std::set<int> unique_labels(labels, labels + g.nodes);
    std::cout << "gpu_csr mode: found " << unique_labels.size() << " communities" << std::endl;
    // Print first 10 labels for sanity check
    std::cout << "First 10 labels: ";
    for (int i = 0; i < std::min(g.nodes, 10); i++) {
        std::cout << labels[i] << " ";
    }
    std::cout << std::endl;
    delete[] labels;
} else {
    cout << "Invalid mode. Use 'cpu' or 'gpu'." << endl;
    return 1;
}

 free(g);
 free_part(part);
                                                                                          
    return 0;
}



