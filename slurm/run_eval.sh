#!/bin/bash --login

#SBATCH --time=00:10:00   # walltime
#SBATCH --ntasks-per-node=8 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=32G   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --qos=cs
#SBATCH -J "run_eval"   # job name
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <=
# ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mamba activate <YOUR_ENV_NAME>
python3 ../../run_eval.py ../../configs/user_configs/<YOUR_CONFIG_HERE>.yaml
