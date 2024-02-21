# Parallel Bellman-Ford Implementation

Parallel Bellman-Ford algorithm implementation using OpenMP and CUDA, for the [Architectures and Platforms for AI](https://www.unibo.it/en/study/phd-professional-masters-specialisation-schools-and-other-programmes/course-unit-catalogue/course-unit/2023/446607) course at the [AI MSc @UNIBO](https://corsi.unibo.it/2cycle/artificial-intelligence).

## Dependencies
The project uses CUDA so a NVIDIA GPU or a compatible one is required to execute this target, as well as the required drivers.

### Software dependencies:
1. GNU Make as the build system.
3. gcc as the C compiler compatible with at least OpenMP 4.5.
2. nvcc as the CUDA compiler.

## Usage
All experiments can be run by simply executing ```./project.sh``` or ```sbatch project.sbatch``` if run on a slurm machine. However, in this case, the file must be changed to accomodate for other slurm machine specifications. All experiments were run on the [HPC Cluster](https://disi.unibo.it/en/department/technical-and-administrative-services/it-services/cluster-hpc) of the [CS Department @UNIBO](https://disi.unibo.it/en).

The targets can be built individually by using the following commands.
```
make [all | gengraph | bfseq | bfomp | bfcuda | clean]

./bin/gengraph.out # Generates random graphs with given parameters
./bin/bfseq.out # Bellman-Ford sequential implementation
./bin/bfomp.out # Bellman-Ford parallel OMP implementation
./bin/bfcuda.out # Bellman-Ford parallel CUDA implementation

```

