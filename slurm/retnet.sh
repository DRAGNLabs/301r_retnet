#!/bin/bash --login

#SBATCH --time=1-00:00:00   # walltime
#SBATCH --ntasks-per-node=1 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=256G   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --qos=cs
#SBATCH -J "retnet"   # job name
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate retnet
python3 ../train_model.py \
    --activation-dropout 0.1 \
    --batch-size 128 \
    --checkpoints \
    --dataset-feature text \
    --dataset-name wikitext \
    --dataset-subset wikitext-2-v1 \
    --device cuda \
    --dropout 0.1 \
    --embed-dim 128 \
    --epochs 10 \
    --ffn-dim 1024 \
    --fsdp \
    --layers 6 \
    --lr 0.001 \
    --model retnet \
    --heads 8 \
    --rand-seed 42 \
    --seq-len 128 \
    --splits 0.7 0.2 0.1 \
    --val-freq 3 \
    --value-embed-dim 128 \
    --vocab-size 100000 \
