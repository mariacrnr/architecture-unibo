// Bellman Ford Algorithm in C

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>

#define INF INT_MAX

void bellmanford(int V, int **graph, int source){
    printf("oi");
    int dist[V];

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
                printf("Graph contains negative-weight cycle\n");
        }
    }

    // Print the distances
    printf("Vertex Distance from Source:\n");
    for (int i = 0; i < V; ++i)
        printf("%d\t\t%d\n", i, dist[i]);
}

int main(){
            
    printf("oia");
    
    FILE *file = fopen("input/small.txt", "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        return 1;
    }

    int source, V;
    int **graph;

    if (fscanf(file, "%d %d", &V, &source) != 2) {
        fprintf(stderr, "Error reading source and vertex values.\n");
        return 1;
    }

    graph = malloc((size_t)V * sizeof(int *));
    for (int i = 0; i < V; ++i) {
        graph[i] = malloc((size_t)V * sizeof(int *));
    }

    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
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

    bellmanford(V, graph, source);

    for (int i = 0; i < V; ++i) {
        free(graph[i]);
    }
    free(graph);

    return 0;
}