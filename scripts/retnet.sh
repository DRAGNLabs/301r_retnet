# This is very small on purpose so you can get code working

python3 ../train_model.py \
    --activation-dropout 0.1 \
    --checkpoint-activations False \
    --dropout 0.0 \
    --embed-dim 8 \
    --ffn-dim 8 \
    --fsdp True \
    --layers 1 \
    --lr 0.01 \
    --model retnet \
    --heads 4 \
    --seq-len 8 \
    --value-embed-dim 8 \
    --vocab-size 28783 \
    --device cpu \
    --epochs 1 \

