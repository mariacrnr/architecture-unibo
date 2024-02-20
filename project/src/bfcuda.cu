
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define INF 1000000
#define BLKDIM 128


int* read_input(char* filename, int *V) {
    char folder[] = "input/";
    char* path = (char*) malloc(strlen(folder) + strlen(filename) + 1);
    strcpy(path, folder);

    FILE *file = fopen(strcat(path, filename), "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
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

void print_output(char* filename, int V, int *distances, int has_negative){
    char folder[] = "output/cuda/";
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

__global__ void bellmanford_kernel(int i, int V, int *graph, int *dist, int *has_changed, int *has_negative){
    int block_index = blockDim.x * blockIdx.x + threadIdx.x;
	int block_inc = blockDim.x * gridDim.x;

	if(block_index < V){
        for(int u = 0 ; u < V ; u ++){
            for(int v = block_index; v < V; v+= block_inc){
                int updated_dist = graph[u * V + v] + dist[u];
                if (graph[u * V + v] < INF && updated_dist < dist[v]){
                    dist[v] = updated_dist;
                    *has_changed = 1;
                }
            }
        }
    }
    __syncthreads();

    if(i == V-1 && has_changed) *has_negative = 1;

}

void bellmanford(int V, int *graph, int source, int *dist, int *has_negative, float *gpu_time){

	int *d_graph, *d_dist;
	int *d_has_changed, h_has_changed;
    int *d_has_negative, h_has_negative;

	cudaMalloc(&d_graph, sizeof(int) * V * V);
	cudaMalloc(&d_dist, sizeof(int) * V);
	cudaMalloc(&d_has_changed, sizeof(int));
    cudaMalloc(&d_has_negative, sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time = 0;

	for(int i = 0 ; i < V; i ++){
		dist[i] = INF;
	}
	dist[source] = 0;

	cudaMemcpy(d_graph, graph, sizeof(int) * V * V, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dist, dist, sizeof(int) * V, cudaMemcpyHostToDevice);

    h_has_negative = 0;

	for(int i = 0; i < V; i++){
		h_has_changed = 0;
		cudaMemcpy(d_has_changed, &h_has_changed, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_has_negative, &h_has_negative, sizeof(int), cudaMemcpyHostToDevice);

        cudaEventRecord(start);
		bellmanford_kernel<<<(V + BLKDIM - 1) / BLKDIM, BLKDIM>>>(i, V, d_graph, d_dist, d_has_changed, d_has_negative);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        *gpu_time += time;

		cudaMemcpy(&h_has_changed, d_has_changed, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(has_negative, d_has_negative, sizeof(int), cudaMemcpyHostToHost);

		if(!h_has_changed || h_has_negative){
			break;
		}

	}

	if(! *has_negative){
		cudaMemcpy(dist, d_dist, sizeof(int) * V, cudaMemcpyDeviceToHost);
	}

	cudaFree(d_graph);
	cudaFree(d_dist);
	cudaFree(d_has_changed);
}

int main(){
    // SOURCE DOS ARGUMENTS
    char filename[] = "negative.txt";

    int source = 0, V, has_negative;
    int *graph = read_input(filename, &V);
    if(graph == NULL) return 1;

    int *dist = (int*) malloc(sizeof(int) * (size_t)V);

    float gpu_time = 0;

    cudaDeviceReset();
    bellmanford(V, graph, source, dist, &has_negative, &gpu_time);
    cudaDeviceSynchronize();

    printf("Elapsed Time: %f milliseconds\n", gpu_time);

    print_output(filename, V, dist, has_negative);

    free(dist);
    free(graph);

    return 0;
}