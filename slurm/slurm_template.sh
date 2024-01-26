#!/bin/bash --login

#SBATCH --time=3-00:00:00   # walltime
#SBATCH --ntasks-per-node=8 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=512G   # memory per CPU core
#SBATCH --gres=gpu:8
#SBATCH --qos=dw87
#SBATCH -J "template_name"   # job name
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
    --data-dir /tmp/data \
    --dataset-name wikitext \
    --datasets-dir /tmp/data/datasets \
    --device cuda \
    --dropout 0.1 \
    --embed-dim 128 \
    --epochs 10 \
    --ffn-dim 1024 \
    --fsdp \
    --heads 8 \
    --layers 6 \
    --lr 0.001 \
    --model retnet \
    --rand-seed 42 \
    --seq-len 128 \
    --tboard-dir /tmp/tboard_logs \
    --tokenizer-folder /tmp/data/tokenizers/BPE_wikitext-2-v1_32768 \
    --val-freq 3 \
    --value-embed-dim 128 \
    --vocab-size 32768 \
