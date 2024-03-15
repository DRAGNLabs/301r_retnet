#!/bin/bash --login

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks-per-node=8 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:8
#SBATCH --qos=dw87
#SBATCH --mem=1000G   # memory per CPU core
#SBATCH -J "train_model"   # job name
#SBATCH --requeue
#SBATCH --signal=SIGHUP@90
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <=
# ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export CODECARBON_LOG_LEVEL="error" # Options: DEBUG, info (default), warning, error, critical

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate 301r
srun python3 \
    ../train_model.py \
    ../configs/user_configs/split_data_random.yaml
