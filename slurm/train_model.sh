#!/bin/bash --login
export restarts=9
#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks-per-node=1 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --qos=cs
#SBATCH --mem=64G   # memory per CPU core
#SBATCH -J "train_model"   # job name
#SBATCH --requeue
#SBATCH --signal=SIGHUP@90
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <=
# ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export CODECARBON_LOG_LEVEL="error" # Options: DEBUG, info (default), warning, error, critical

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate <YOUR_ENV_NAME>
srun python3 \
    ../../src/train_model.py \
    ../../configs/user_configs/<YOUR_CONFIG_HERE>.yaml
