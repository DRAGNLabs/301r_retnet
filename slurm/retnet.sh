#!/bin/bash --login

#SBATCH --time=1-00:00:00   # walltime
#SBATCH --ntasks-per-node=8 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=256G   # memory per CPU core
#SBATCH --gres=gpu:8
#SBATCH --qos=dw87
#SBATCH -J "retnet"   # job name
#SBATCH --output=%x_%j.out

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate retnet
srun python3 ../train_model_lightning.py \
    --activation-dropout 0.1 \
    --batch-size 8 \
    --checkpoints \
    --data-dir /home/jo288/compute/retnet/data \
    --dataset-dir /home/jo288/compute/retnet/data/dataset\
    --dataset-feature text \
    --dataset-name wikitext \
    --dataset-subset wikitext-2-v1 \
    --device cuda \
    --dropout 0.1 \
    --embed-dim 64 \
    --epochs 1 \
    --ffn-dim 512 \
    --fsdp \
    --heads 8 \
    --layers 6 \
    --lr 0.001 \
    --model retnet \
    --rand-seed 42 \
    --seq-len 128 \
    --splits 0.7 0.2 0.1 \
    --tboard-dir /tmp/tboard_logs \
    --val-freq 1 \
    --value-embed-dim 128 \
    --vocab-size 20000 \
    --tokenizer-folder /home/jo288/compute/retnet/data/tokenizers/wikitext \
    --num-devices 8 \

