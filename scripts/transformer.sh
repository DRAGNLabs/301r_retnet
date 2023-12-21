python3 ../train_model.py \
    --activation-dropout 0.1 \
    --dropout 0.1 \
    --embed-dim 10 \
    --ffn-dim 10 \
    --fsdp \
    --layers 2 \
    --lr 0.0001 \
    --model transformer \
    --heads 1 \
    --seq-len 30 \
    --value-embed-dim 10 \
    --vocab-size 1000 \
    --rand-seed 42 \
