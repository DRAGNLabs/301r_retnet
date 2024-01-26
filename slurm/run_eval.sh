#!/bin/bash --login

#SBATCH --time=00:10:00   # walltime
#SBATCH --ntasks-per-node=8 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=32G   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --qos=cs
#SBATCH -J "lm_eval"   # job name
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# Make sure to set these
MODEL_PATH_DIR=/tmp/models/retnet
TOKENIZER_PATH_DIR=/grphome/grp_retnet/compute/tokenizers/wikitext
TASKS=winogrande


# BEFORE RUNNING THIS SCRIPT ON THE COMPUTE NODE: Make sure to run the run_eval.py script locally to download and cache the correct datasets and benchmarks.
# You do not need to let it run to completion, just until the datasets and benchmarks are downloaded and cached.

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate retnet
python ../run_eval.py --model-path-dir ${MODEL_PATH_DIR} --tokenizer-path-dir ${TOKENIZER_PATH_DIR} --tasks ${TASKS}
