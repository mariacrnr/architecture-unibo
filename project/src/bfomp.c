// Bellman Ford Algorithm in C

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <time.h>

#define INF INT_MAX

int** read_input(char* filename, int *source, int *V) {
    char folder[] = "input/";
    char *path = malloc(strlen(folder) + strlen(filename) + 1);
    strcpy(path, folder);

    FILE *file = fopen(strcat(path, filename), "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        return NULL;
    }

    if (fscanf(file, "%d %d", V, source) != 2) {
        fprintf(stderr, "Error reading source and vertex values.\n");
        fclose(file);
        return NULL;
    }

    int** graph = malloc((size_t)(*V) * sizeof(int *));
    if (graph == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        fclose(file);
        return NULL;
    }

    for (int i = 0; i < *V; ++i) {
        graph[i] = malloc((size_t)(*V) * sizeof(int *));
        if (graph[i] == NULL) {
            fprintf(stderr, "Memory allocation error.\n");
            fclose(file);
            for (int j = 0; j < i; ++j) {
                free(graph[j]);
            }
            free(graph);
            return NULL;
        }
    }

    for (int i = 0; i < *V; ++i) {
        for (int j = 0; j < *V; ++j) {
            char token[10]; // Assuming "INF" won't be longer than 10 characters
            if (fscanf(file, "%s", token) == 1) {
                if (strcmp(token, "INF") == 0) {
                    graph[i][j] = INF;
                } else {
                    graph[i][j] = atoi(token);
                }
            }
        }
    }

    fclose(file);

    return graph;
}

void print_output(char* filename, int V, int *distances){
    char folder[] = "output/";
    char *path = malloc(strlen(folder) + strlen(filename) + 1);
    strcpy(path, folder);

    FILE *file = fopen(strcat(path, filename), "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        return;
    }

    for (int i = 0; i < V; ++i) {
        fprintf(file, "%d\t\t%d\n", i, distances[i]);
    }
    fprintf(file, "\n");

    fclose(file);
}

void bellmanford(int V, int **graph, int source, int *dist){

    // Initialize distances from source to all other vertices as INFINITE
    for (int i = 0; i < V; ++i)
        dist[i] = INF;
    dist[source] = 0;


    // Relax all edges |V| - 1 times
    for (int i = 0; i < V - 1; ++i) {
        for (int u = 0; u < V; ++u) {
            for (int v = 0; v < V; ++v) {
                if (graph[u][v] != INF && dist[u] + graph[u][v] < dist[v])
                    dist[v] = dist[u] + graph[u][v];
            }
        }
    }

    // Check for negative-weight cycles
    for (int u = 0; u < V; ++u) {
        for (int v = 0; v < V; ++v) {
            if (graph[u][v] != INF && dist[u] + graph[u][v] < dist[v])
                printf("Graph contains negative-weight cycle!\n");
        }
    }
}

int main(){

    char filename[] = "small.txt";

    int source, V;
    int **graph = read_input(filename, &source, &V);
    if(graph == NULL) return 1;

    int dist[V];

    clock_t begin = clock();
    bellmanford(V, graph, source, dist);
    clock_t end = clock();

    printf("Runtime: %f seconds\n", (double)(end - begin)/ CLOCKS_PER_SEC);

    print_output(filename, V, dist);

    for (int i = 0; i < V; ++i) {
        free(graph[i]);
    }
    free(graph);

    return 0;
}