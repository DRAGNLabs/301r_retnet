#!/bin/bash --login

#SBATCH --time=1-00:00:00   # walltime
#SBATCH --ntasks-per-node=1 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=128G   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --qos=cs
#SBATCH -J "gridsearch"   # job name
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate retnet
srun python3 ../grid_search.py \
    --data-dir /grphome/grp_retnet/compute/data \
    --dataset-dir /grphome/grp_retnet/compute/data \
    --dataset-feature text \
    --dataset-name wikitext \
    --dataset-subset wikitext-2-v1 \
    --train-data /grphome/grp_retnet/compute/data/wikitext/tokenized/train.parquet \
    --validation-data /grphome/grp_retnet/compute/data/wikitext/tokenized/validation.parquet \
    --test-data /grphome/grp_retnet/compute/data/wikitext/tokenized/test.parquet \
    --tokenizer-folder /grphome/grp_retnet/compute/tokenizers/wikitext \
    
