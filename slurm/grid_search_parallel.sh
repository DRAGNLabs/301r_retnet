#!/bin/bash --login

#SBATCH --time=00:10:00   # walltime
#SBATCH --ntasks-per-node=2 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=128G   # memory per CPU core
#SBATCH --gres=gpu:2
#SBATCH --qos=cs
#SBATCH -J "gridsearch"   # job name
#SBATCH --output=./data/out_files/retnet/%x_%j_%a.out    # you can change the path to the data_dir/model_type here
#SBATCH --array=0

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi
mamba activate new_retnet

# check that the model type is specified
if [ -z "$1" ]
then
    echo "Please specify a model type (transformer or retnet): $0 [model_type]"
    exit 1
fi

model_type=$1

# set up grid search across the following hyper parameters (in order):
# learning_rates = [0.01, 0.001, 0.0001]
# embed_dims = [768, 1024, 1280]
# batch_sizes = [16, 32, 64]

LEARNING_RATES=(0.01 0.001 0.0001)
EMBED_DIMS=(768 1024 1280)
BATCH_SIZES=(16 32 64)

# get the index of the current job
INDEX=$SLURM_ARRAY_TASK_ID

# get the learning rate
LR_INDEX=$((INDEX % 3))
LR=${LEARNING_RATES[$LR_INDEX]}
echo "LR: $LR"

# get the embed dim
ED_INDEX=$(((INDEX / 3) % 3))
ED=${EMBED_DIMS[$ED_INDEX]}
echo "ED: $ED"

# get the batch size
BS_INDEX=$(((INDEX / 9) % 3))
BS=${BATCH_SIZES[$BS_INDEX]}
echo "BS: $BS"


srun python3 ../train_model_lightning.py \
    --activation-dropout 0.1 \
    --batch-size ${BS} \
    --checkpoints \
    --data-dir ./data/${model_type} \
    --dataset-dir /grphome/grp_retnet/compute/data \
    --dataset-feature text \
    --dataset-name wikitext \
    --dataset-subset wikitext-2-v1 \
    --device cuda \
    --dropout 0.1 \
    --embed-dim ${ED} \
    --epochs 1 \
    --ffn-dim 512 \
    --fsdp \
    --heads 8 \
    --layers 6 \
    --lr ${LR} \
    --model ${model_type} \
    --rand-seed 42 \
    --seq-len 128 \
    --splits 0.7 0.2 0.1 \
    --tboard-dir /tmp/tboard_logs \
    --val-freq 1 \
    --value-embed-dim 128 \
    --vocab-size 4000 \
    --tokenizer-folder /grphome/grp_retnet/compute/tokenizers/wikitext \
    --num-devices 2 \
    --train-data /grphome/grp_retnet/compute/data/wikitext/tokenized/train.parquet \
    --validation-data /grphome/grp_retnet/compute/data/wikitext/tokenized/validation.parquet \
    --test-data /grphome/grp_retnet/compute/data/wikitext/tokenized/test.parquet \
    --grid-search-out-file ./test.out
    # --data-dir /grphome/grp_retnet/compute/data \