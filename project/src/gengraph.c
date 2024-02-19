#include<stdio.h>
#include<stdlib.h>
#include <time.h>

#define INF 1000000

void generate_random_graph(int vertices, int **graph, int negative_edges, int density, int weight_range) {
    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < vertices; j++) {
            if (i == j) { // No self-loops
                graph[i][j] = 0;
            } else {
                int randNum = negative_edges? rand() % (weight_range * 2 + 1) - weight_range : rand() % (weight_range + 1);
                if (randNum < density) {
                    graph[i][j] = randNum;
                } else {
                    graph[i][j] = INF;
                }
            }
        }
    }
}

void write_graph(int vertices, int **graph, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        exit(1);
    }

    fprintf(file, "%d \n", vertices);
    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < vertices; j++) {
            if (graph[i][j] == INF) fprintf(file, "INF ");
            else fprintf(file, "%d ", graph[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main(){
    int num_vertices = 1000;
    int weight_range = 100;
    int negative_edges = 0;
    int density = 30;

    int **graph = (int **)malloc(num_vertices * sizeof(int *));
    if (graph == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }
    for (int i = 0; i < num_vertices; i++) {
        graph[i] = (int *)malloc(num_vertices * sizeof(int));
        if (graph[i] == NULL) {
            printf("Memory allocation failed.\n");
            return 1;
        }
    }

    srand(55); //Fixed value for reproducibility

    generate_random_graph(num_vertices, graph, negative_edges, density, weight_range);

    const char *filename = "graph.txt";
    write_graph(num_vertices, graph, filename);
    printf("Graph written to file '%s'.\n", filename);

    return 0;
}