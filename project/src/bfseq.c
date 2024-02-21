// bfseq.c
// Implements the sequential version of the Bellman-Ford Algorithm. 
// Not used in experiments, used only for development to check if the parallel versions produce correct results.

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <time.h>

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
                    graph[i * (*V) + j] = INF; // Save graph as 1D instead of 2D to save memory
                } else {
                    graph[i * (*V) + j] = atoi(token);
                }
            }
        }
    }

    fclose(file);
    free(path);

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
 */
void write_output(char* filename, int V, int *distances, int has_negative){
    char folder[] = "output/test/";
    char* path = (char*) malloc(strlen(folder) + strlen(filename) + 1);
    strcpy(path, folder);

    char sfilename[256];
    sprintf(sfilename, "%s.txt", filename);

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
    free(path);
}


/**
 * Implements the sequential version of the Bellman-Ford algorithm to find shortest paths from a source vertex to all the others.
 *
 * @param V Number of vertices in the graph.
 * @param graph Pointer to the array representing the graph.
 * @param source Source vertex from which shortest paths are computed.
 * @param dist Array to store the distances from the source vertex.
 * @param has_negative Pointer to a flag indicating whether the graph contains a negative cycle.
 */
void bellmanford(int V, int *graph, int source, int *dist, int *has_negative){

    // Initialize distances from source to all other vertices as INF
    for (int i = 0; i < V; i++)
        dist[i] = INF;
    dist[source] = 0;

    // Relax all edges |V| - 1 times
    for (int i = 0; i < V - 1; i++) {
        for (int u = 0; u < V; u++) {
            for (int v = 0; v < V; v++) {
                if (graph[u * V + v] != INF && dist[u] + graph[u * V + v] < dist[v]){
                    dist[v] = dist[u] + graph[u * V + v];
                }
            }
        }
    }

    // Check for negative-weight cycles
    for (int u = 0; u < V; u++) {
        for (int v = 0; v < V; v++) {
            if (graph[u * V + v] != INF && dist[u] + graph[u * V + v] < dist[v])
                *has_negative = 1;
        }
    }
}

int main(int argc, char **argv){
    if (argc != 3) {
        printf("Usage: %s source_vertex filename \n", argv[0]);
        return 1;
    }

    int source = atoi(argv[1]);
    char *filename = argv[2];

    int V, has_negative = 0;
    int *graph = read_input(filename, &V);
    if(graph == NULL) return 1;

    int *dist = (int*) malloc(sizeof(int) * (size_t)V);

    clock_t begin = clock();
    bellmanford(V, graph, source, dist, &has_negative);
    clock_t end = clock();

    printf("Elapsed Time: %f milliseconds\n", ((double)(end - begin)* 1000)/ CLOCKS_PER_SEC);

    write_output(filename, V, dist, has_negative);

    free(dist);
    free(graph);

    return 0;
}