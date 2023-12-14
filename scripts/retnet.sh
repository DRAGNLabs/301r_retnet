python3 ../train_model.py \
    --activation-dropout 0.0 \
    --checkpoint-activations False \
    --dropout 0.0 \
    --embed-dim 32 \
    --ffn-dim 64 \
    --fsdp True \
    --layers 1 \
    --lr 0.001 \
    --model retnet \
    --heads 8 \
    --seq-len 128 \
    --value-embed-dim 32 \
    --vocab-size 28783 \
    --device cpu \
    --epochs 10 \
    --batch-size 128 \

