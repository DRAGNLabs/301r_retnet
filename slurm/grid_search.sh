#!/bin/bash --login

#SBATCH --time=23:00:00   # walltime
#SBATCH --ntasks-per-node=4 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:a100:4
#SBATCH --qos=cs
#SBATCH --mem=64G   # memory per CPU core
#SBATCH -J "grid_search"   # job name
#SBATCH --requeue
#SBATCH --signal=SIGHUP@90
#SBATCH --output=%x_%j.out
#SBATCH --array=0-15%4  # '%n' limits the # of jobs running at once to 'n'

# Set the max number of threads to use for programs using OpenMP. Should be <=
# ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

mamba activate <YOUR_ENV_NAME>

srun python3 ../../src/grid_search.py \
    ../../configs/user_configs/<YOUR_CONFIG_HERE>.yaml \
    $SLURM_ARRAY_TASK_ID