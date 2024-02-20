#include<stdio.h>
#include<stdlib.h>
#include <time.h>
#include <string.h>

#define INF 1000000

void generate_random_graph(int vertices, int *graph, int is_negative, double edge_probability, int max_weight) {
    srand(42);
    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < vertices; j++) {
            if (i == j) { // No self-loops
                graph[i * vertices + j] = 0;
            } else {
                double rand_n = (double) rand() / RAND_MAX;
                if (rand_n <= edge_probability) {
                    graph[i * vertices + j] = is_negative? rand() % (max_weight * 2 + 1) - max_weight : rand() % max_weight + 1;
                } else {
                    graph[i * vertices + j] = INF;
                }
            }
        }
    }
}

void write_graph(int vertices, int *graph, const char *filename) {
    char folder[] = "input/";
    char* path = (char*) malloc(strlen(folder) + strlen(filename) + 1);
    strcpy(path, folder);

    FILE *file = fopen(strcat(path, filename), "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        exit(1);
    }

    fprintf(file, "%d 0\n", vertices);
    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < vertices; j++) {
            if (graph[i * vertices + j] == INF) fprintf(file, "INF ");
            else fprintf(file, "%d ", graph[i * vertices + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main(){
    int num_vertices = 1000;
    int max_weight = 2000;
    int is_negative = 1;
    double edge_probability = 0.3;

    int *graph = (int*) malloc((size_t)(num_vertices) * (size_t)(num_vertices) * sizeof(int *));
    if (graph == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    generate_random_graph(num_vertices, graph, is_negative, edge_probability, max_weight);

    const char *filename = "graph.txt";
    write_graph(num_vertices, graph, filename);
    printf("Graph written to file '%s'.\n", filename);

    return 0;
}