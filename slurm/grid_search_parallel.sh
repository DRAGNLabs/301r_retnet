#!/bin/bash --login

#SBATCH --time=05:00:00   # walltime
#SBATCH --ntasks-per-node=8 # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=128G   # memory per CPU core
#SBATCH --gres=gpu:8
#SBATCH --qos=cs
#SBATCH -J "gridsearch"   # job name
#SBATCH --output=/grphome/grp_retnet/compute/data/data_run/out_files/retnet/%x_%j_%a.out    # you can change the path to the data_dir/model_type here
#SBATCH --array=0-71

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
# learning_rates = [0.001, 0.0005, 0.0001]
# embed_dims = [768, 1024, 1280]
# ffn_dim = [1024, 2048]
# heads = [4, 8]
# seq_len = [256, 512]

LEARNING_RATES=(0.001 0.0005 0.0001)
EMBED_DIMS=(768 1024 1280)
FFN_DIMS=(1024 2048)
HEADS=(4 8)
SEQ_LEN=(256 512)

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

# get the ffn dim
FFN_INDEX=$(((INDEX / 9) % 2))
FFN=${FFN_DIMS[$FFN_INDEX]}
echo "FFN: $FFN"

# get the heads
HEADS_INDEX=$(((INDEX / 18) % 2))
HEADS=${HEADS[$HEADS_INDEX]}
echo "HEADS: $HEADS"

# get the seq len
SEQ_LEN_INDEX=$(((INDEX / 36) % 2))
SEQ_LEN=${SEQ_LEN[$SEQ_LEN_INDEX]}
echo "SEQ_LEN: $SEQ_LEN"


srun python3 ../train_model_lightning.py \
    --activation-dropout 0.1 \
    --batch-size 32 \
    --checkpoints \
    --data-dir /grphome/grp_retnet/compute/data/data_run \
    --dataset-dir /grphome/grp_retnet/compute/data \
    --dataset-feature text \
    --dataset-name wikitext \
    --dataset-subset wikitext-103-raw-v1 \
    --device cuda \
    --dropout 0.1 \
    --embed-dim ${ED} \
    --epochs 5 \
    --ffn-dim ${FFN} \
    --fsdp \
    --heads ${HEADS} \
    --layers 6 \
    --lr ${LR} \
    --model ${model_type} \
    --rand-seed 42 \
    --seq-len ${SEQ_LEN} \
    --splits 0.7 0.2 0.1 \
    --tboard-dir /tmp/tboard_logs \
    --val-freq 1 \
    --value-embed-dim 128 \
    --vocab-size 32768 \
    --tokenizer-folder /grphome/grp_retnet/compute/tokenizers/wikitext \
    --num-devices 8 \
    --train-data /grphome/grp_retnet/compute/data/wikitext/tokenized/train.parquet \
    --validation-data /grphome/grp_retnet/compute/data/wikitext/tokenized/validation.parquet \
    --test-data /grphome/grp_retnet/compute/data/wikitext/tokenized/test.parquet \
    --grid-search-out-file /grphome/grp_retnet/compute/data/data_run/out_files/${model_type}_grid_search_results.csv \