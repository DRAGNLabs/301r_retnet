#!/bin/bash --login

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks-per-node=1# number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=64G   # memory per CPU core
#SBATCH -J "tokenize_data"   # job name
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <=
# ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate <YOUR_ENV_NAME>
python3 \
    ../../src/tokenize_data.py \
    ../../configs/user_configs/<YOUR_CONFIG_HERE>.yaml \
    --split <YOUR_SPLIT_HERE> # train, validation, or test. You can start a job for each split.
