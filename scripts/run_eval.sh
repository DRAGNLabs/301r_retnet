# This script is to run the EleutherAI Model Evaluation Harness, for full documentation see their github: https://github.com/EleutherAI/lm-evaluation-harness


# Get configuration and weight file path from command line argument
CONFIG_PATH=$1

lm_eval --model hf \
    --model_args pretrained=${CONFIG_PATH} \
    --tasks hellaswag \
    --device cpu \
    --batch_size 8 \
    --verbosity DEBUG


