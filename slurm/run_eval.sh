#!/bin/bash --login

#SBATCH --time=00:20:00   # walltime
#SBATCH --ntasks-per-node=8 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=512G   # memory per CPU core
#SBATCH --gres=gpu:1
#SBATCH --qos=dw87
#SBATCH -J "lm_eval"   # job name
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate retnet
python run_eval.py --model_path_dir ./retnet --tokenizer_path_dir /home/datingey/fsl_groups/grp_retnet/compute/tokenizers/wikitext --tasks winogrande
