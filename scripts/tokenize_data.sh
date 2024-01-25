python3 ../tokenize_data.py \
    --data-dir /tmp/data \
    --dataset-name wikitext \
    --dataset-subset wikitext-2-v1 \
    --datasets-dir /tmp/data/datasets \
    --seq-len 128 \
    --splits 0.7 0.2 0.1 \
    --text-feature text \
    --tokenized-data-name wikitext-2-v1-tokenized \
    --tokenizer-folder /tmp/data/tokenizers/BPE_wikitext-2-v1_32768 \
