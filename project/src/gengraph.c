// gengraph.c
// Generates random graphs used for testing the parallel implementations of the algorithms.

#include<stdio.h>
#include<stdlib.h>
#include <time.h>
#include <string.h>

#define INF 1000000

/**
 * Generates a random graph with given parameters.
 *
 * @param vertices Number of vertices in the graph.
 * @param graph Pointer to the array representing the graph.
 * @param is_negative Flag indicating whether the graph can have negative weights.
 * @param edge_probability Probability of having an edge between two vertices.
 * @param max_weight Maximum weight of edges.
 */
void generate_random_graph(int vertices, int *graph, int is_negative, double edge_probability, int max_weight ) {
    srand(42); // Same seed for reproducibility
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


/**
 * Write the generated graph to a file. Each line of the file represents a row in the adjacency matrix of the graph.
 *
 * @param vertices Number of vertices in the graph.
 * @param graph Pointer to the array representing the graph.
 * @param filename Name of the file to write the graph to.
 */
void write_graph(int vertices, int *graph, const char *filename) {
    char folder[] = "input/test/";
    char* path = (char*) malloc(strlen(folder) + strlen(filename) + 1);
    strcpy(path, folder);

    FILE *file = fopen(strcat(strcat(path, filename), ".txt"), "w");
    if (file == NULL) {
        printf("Error opening file.\n");
        exit(1);
    }

    fprintf(file, "%d\n", vertices);
    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < vertices; j++) {
            if (graph[i * vertices + j] == INF) fprintf(file, "INF ");
            else fprintf(file, "%d ", graph[i * vertices + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main(int argc, char **argv){
    if (argc != 6) {
        printf("Usage: %s num_vertices max_weight is_negative edge_probability filename.txt\n", argv[0]);
        return 1;
    }

    int num_vertices = atoi(argv[1]);
    int max_weight = atoi(argv[2]);
    int is_negative = atoi(argv[3]);
    double edge_probability = atof(argv[4]) / 100;
    const char *filename = argv[5];

    int *graph = (int*) malloc((size_t)(num_vertices) * (size_t)(num_vertices) * sizeof(int *));
    if (graph == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    generate_random_graph(num_vertices, graph, is_negative, edge_probability, max_weight);

    write_graph(num_vertices, graph, filename);
    printf("Graph generated and written to file '%s.txt'!\n", filename);

    return 0;
}