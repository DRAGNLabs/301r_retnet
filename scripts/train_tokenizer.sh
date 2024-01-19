python3 ../train_tokenizer.py \
--tokenizer_folder grp_home/grp_retnet/compute/data/tokenizers \
--dataset_name wikitext \
--seq_len 256 \
--vocab_size 20000 \
--dataset_dir grp_home/grp_retnet/compute/data/datasets \
--dataset_subset wikitext-2-v1 \
--text_feature text \
--splits 0.7 0.2 0.1 \
--rand_seed 42 \