// Bellman Ford Algorithm in C

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <time.h>
#include <omp.h>


#define INF 1000000

int* read_input(char* filename, int *source, int *V) {
    char folder[] = "input/";
    char* path = (char*) malloc(strlen(folder) + strlen(filename) + 1);
    strcpy(path, folder);

    FILE *file = fopen(strcat(path, filename), "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        return NULL;
    }

    if (fscanf(file, "%d %d", V, source) != 2) {
        fprintf(stderr, "Error reading source and vertex count values.\n");
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

void print_output(char* filename, int V, int *distances, int has_negative){
    char folder[] = "output/omp/";
    char* path = (char*) malloc(strlen(folder) + strlen(filename) + 1);
    strcpy(path, folder);

    FILE *file = fopen(strcat(path, filename), "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
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

void bellmanford(int V, int *graph, int source, int *dist, int threads, int *has_negative){

    // Initialize thread distribution
    int thread_start[threads], thread_end[threads];
    int q = V / threads, r = V % threads;

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
                        thread_changed[rank] = 1;
                    }
                }
            }

            #pragma omp barrier         
            #pragma omp single
            {
                dist_changed = 0;
                for(int rank_n = 0; rank_n < threads; rank_n++){
                    if(thread_changed[rank_n] && (i == V - 1)){
                        *has_negative = 1;
                        break;
                    } 
                    dist_changed |= thread_changed[rank_n];
                }
            }

            if(!dist_changed || *has_negative) break;
        }
    }
}

int main(){

    char filename[] = "graph.txt";

    int threads = atoi(getenv("OMP_NUM_THREADS"));

    int source, V, has_negative = 0;
    int *graph = read_input(filename, &source, &V);
    if(graph == NULL) return 1;

    int *dist = (int*) malloc(sizeof(int) * (size_t)V);

    double tstart = omp_get_wtime();
    bellmanford(V, graph, source, dist, threads, &has_negative);
    double tstop = omp_get_wtime();

    printf("Elapsed Time: %f milliseconds\n", (tstop-tstart) * 1000);

    print_output(filename, V, dist, has_negative);

    free(dist);
    free(graph);

    return 0;
}