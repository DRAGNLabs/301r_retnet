#!/bin/bash

#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4096M   # memory per CPU core


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export TORCHDYNAMO_VERBOSE=1

python3 ../train_model.py \
    --activation-dropout 0.0 \
    --dropout 0.0 \
    --checkpoints \
    --embed-dim 32 \
    --ffn-dim 64 \
    --fsdp \
    --layers 1 \
    --lr 0.001 \
    --model retnet \
    --heads 8 \
    --seq-len 128 \
    --value-embed-dim 32 \
    --vocab-size 28783 \
    --device cpu \
    --epochs 1 \
    --batch-size 128 \
    --rand-seed 42 \
