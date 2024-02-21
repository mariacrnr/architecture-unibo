// bfomp.c
// Implements the OMP parallel version of the Bellman-Ford Algorithm.

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <time.h>
#include <omp.h>


#define INF 1000000

/**
 * Read the input graph from a file. The file should have the number of vertices on the first line followed by the adjacency matrix of the graph.
 *
 * @param filename Name of the file containing the graph.
 * @param V Pointer to store the number of vertices read from the file.
 * @return Pointer to the array representing the graph if successful, otherwise NULL.
 */
int* read_input(char* filename, int *V) {
    char folder[] = "input/test/";
    char* path = (char*) malloc(strlen(folder) + strlen(filename) + 1);
    strcpy(path, folder);

    FILE *file = fopen(strcat(strcat(path, filename), ".txt"), "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening input file.\n");
        return NULL;
    }

    if (fscanf(file, "%d", V) != 1) {
        fprintf(stderr, "Error reading vertex count.\n");
        fclose(file);
        return NULL;
    }

    int* graph = (int*) malloc((size_t)(*V) * (size_t)(*V) * sizeof(int *));
    if (graph == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < *V; i++) {
        for (int j = 0; j < *V; j++) {
            char token[10]; 
            if (fscanf(file, "%s", token) == 1) {
                if (strcmp(token, "INF") == 0) {
                    graph[i * (*V) + j] = INF;
                } else {
                    graph[i * (*V) + j] = atoi(token);
                }
            }
        }
    }

    fclose(file);

    return graph;
}

/**
 * Write the output distances from the Bellman-Ford algorithm to a file. If the graph contains a negative cycle, 
 * it writes a message indicating the presence of a negative cycle.
 * 
 * @param filename Name of the output file.
 * @param V Number of vertices in the graph.
 * @param distances Array containing the distances from the source vertex.
 * @param has_negative Flag indicating whether the graph contains a negative cycle.
 * @param threads Number of threads used in the computation.
 */
void write_output(char* filename, int V, int *distances, int has_negative, int threads){
    char folder[] = "output/omp/";
    char* path = (char*) malloc(strlen(folder) + strlen(filename) + 1);
    strcpy(path, folder);

    char sfilename[256];
    sprintf(sfilename, "%s_%d.txt", filename, threads);

    FILE *file = fopen(strcat(path, sfilename), "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening output file.\n");
        return;
    }

    if(has_negative){
        fprintf(file, "Graph contains negative cycle!!");
    } else {
        for (int i = 0; i < V; i++) {
            fprintf(file, "%d\t\t%d\n", i, distances[i]);
        }
    }
    fprintf(file, "\n");

    fclose(file);
}

/**
 * Implements the OMP parallel version of the Bellman-Ford algorithm to find shortest paths from a source vertex. 
 *
 * @param V Number of vertices in the graph.
 * @param graph Pointer to the array representing the graph.
 * @param source Source vertex from which shortest paths are computed.
 * @param dist Array to store the distances from the source vertex.
 * @param threads Number of threads to use for parallelization.
 * @param has_negative Pointer to a flag indicating whether the graph contains a negative cycle.
 */
void bellmanford(int V, int *graph, int source, int *dist, int threads, int *has_negative){

    // Initialize thread distribution
    int thread_start[threads], thread_end[threads];
    int q = V / threads, r = V % threads;

    // Distributes the vertices for the threads
    #pragma omp parallel for
    for (int i = 0; i < threads; i++){
        thread_start[i] = q * i + (i < r ? i : r);
        thread_end[i] = thread_start[i] + q + (i < r ? 1 : 0);
    }

    // Initialize distances from source to all other vertices as INF
    #pragma omp parallel for
    for (int i = 0; i < V; i++){
        dist[i] = INF;
    }
    dist[source] = 0;

    int dist_changed = 0;
    int thread_changed[threads];

    // Paralellizes main loop
    #pragma omp parallel
    {
        int rank = omp_get_thread_num();
        int my_start = thread_start[rank];
        int my_end = thread_end[rank];

        for (int i = 0; i < V; i++) {
            thread_changed[rank] = 0;

            for (int u = 0; u < V; u++) {
                for (int v = my_start; v < my_end; v++) {
                    int updated_dist = graph[u * V + v] + dist[u];
                    if (graph[u * V + v] < INF && updated_dist < dist[v]){
                        dist[v] = updated_dist;
                        thread_changed[rank] = 1; // True when the distance has changed
                    }
                }
            }

            #pragma omp barrier  // Threads must sync       
            #pragma omp single
            {
                dist_changed = 0;
                for(int rank_n = 0; rank_n < threads; rank_n++){
                    if(thread_changed[rank_n] && (i == V - 1)){ 
                        *has_negative = 1; // Graph contains negative cycle if a distance has decreased in the last iteration
                        break;
                    } 
                    dist_changed |= thread_changed[rank_n]; // Verifies if any thread changed distances
                }
            }

            if(!dist_changed || *has_negative) break; // Return if there is a negative cycle or if the last cycle didn't provide any changes in distance

        }
    }
}

int main(int argc, char **argv){


    if (argc != 3) {
        printf("Usage: OMP_NUM_THREADS=num_threads %s source_vertex filename \n", argv[0]);
        return 1;
    }

    int threads = atoi(getenv("OMP_NUM_THREADS"));

    int source = atoi(argv[1]);
    char *filename = argv[2];

    int V, has_negative = 0;
    int *graph = read_input(filename, &V);
    if(graph == NULL) return 1;

    int *dist = (int*) malloc(sizeof(int) * (size_t)V);

    double tstart = omp_get_wtime();
    bellmanford(V, graph, source, dist, threads, &has_negative);
    double tstop = omp_get_wtime();

    printf("%d Vertices -> Elapsed Time: %f milliseconds\n", V, (tstop-tstart) * 1000);

    write_output(filename, V, dist, has_negative, threads);

    free(dist);
    free(graph);

    return 0;
}