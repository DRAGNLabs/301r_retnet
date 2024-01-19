<<<<<<< HEAD
python3 ../train_model.py \
    --activation-dropout 0.1 \
    --batch-size 128 \
    --checkpoints \
    --dataset-feature text \
    --dataset-name c4 \
    --dataset-subset en \
    --device cuda \
    --dropout 0.1 \
    --embed-dim 128 \
    --epochs 10 \
    --ffn-dim 1024 \
    --fsdp \
    --layers 6 \
    --lr 0.001 \
    --model retnet \
    --heads 8 \
    --seq-len 128 \
    --value-embed-dim 32 \
    --vocab-size 28783 \
    --device cpu \
    --epochs 10 \
    --batch-size 128 \
    --rand-seed 42 \
    --seq-len 128 \
    --splits 0.7 0.2 0.1 \
    --val-freq 3 \
    --value-embed-dim 128 \
    --vocab-size 100000 \
=======
python3 ../train_model.py \
    --activation-dropout 0.1 \
    --batch-size 128 \
    --checkpoints \
    --data-dir /tmp/data \
    --dataset-dir /tmp/data/datasets \
    --dataset-feature text \
    --dataset-name wikitext \
    --dataset-subset wikitext-2-v1 \
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
    --splits 0.7 0.2 0.1 \
    --tboard-dir /tmp/tboard_logs \
    --val-freq 3 \
    --value-embed-dim 128 \
    --vocab-size 100000
>>>>>>> origin
