python3 ../train_tokenizer.py \
    --dataset-name wikitext \
    --dataset-subset wikitext-2-v1 \
    --datasets-dir /tmp/data/datasets \
    --rand-seed 42 \
    --seq-len 256 \
    --splits 0.7 0.2 0.1 \
    --text-feature text \
    --tokenizer-folder /tmp/data/tokenizers/BPE_wikitext-2-v1_32768 \
    --vocab-size 32768 \
