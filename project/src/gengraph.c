#include<stdio.h>
#include<stdlib.h>
#include <time.h>
#include <string.h>

#define INF 1000000

void generate_random_graph(int vertices, int *graph, int is_negative, double edge_probability, int max_weight ) {
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
    char folder[] = "input/test/";
    char* path = (char*) malloc(strlen(folder) + strlen(filename) + 1);
    strcpy(path, folder);

    FILE *file = fopen(strcat(path, filename), "w");
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

    // char * sign = is_negative? " and negative " : " ";
    // printf("Generating graph '%s' with %d vertices and %d%% edge probability with positive%sedge weights ranging to %d.\n", filename, num_vertices, (int)(edge_probability * 100), sign, max_weight);

    generate_random_graph(num_vertices, graph, is_negative, edge_probability, max_weight);

    write_graph(num_vertices, graph, filename);
    printf("Graph generated and written to file '%s'!\n", filename);

    return 0;
}