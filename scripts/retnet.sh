python3 ../train_model.py \
    --activation-dropout 0.1 \
    --batch-size 128 \
    --checkpoints \
    --dataset-dir /tmp/data \
    --dataset-feature text \
    --dataset-name c4 \
    --dataset-subset en \
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
    --models-dir /tmp/models \
    --rand-seed 42 \
    --seq-len 128 \
    --splits 0.7 0.2 0.1 \
    --tboard-dir /tmp/tboard_logs \
    --val-freq 3 \
    --value-embed-dim 128 \
    --vocab-size 100000
# Specify where to save weights. Should be a directory that doesn't exist.
# Commented out to prevent models overwritting each other's weights.
#   --weights-dir /tmp/weights \
