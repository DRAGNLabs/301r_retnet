#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --qos=dw87
#SBATCH --mem-per-cpu=4096M   # memory per CPU core


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export TORCHDYNAMO_VERBOSE=1

python3 ../grid_search.py \
    --dataset-name wikitext \
    --dataset-subset wikitext-2-v1 \
    --dataset-dir /tmp/data/datasets \
    --data-dir /tmp/data \
    --dataset-feature text \
