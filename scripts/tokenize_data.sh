python3 ../tokenize_data.py \
--tokenized_data_name wikitext-2-v1-tokenized \
--tokenized_data_folder /home/searlejj/fsl_groups/grp_retnet/compute/data\
--tokenizer_folder /home/searlejj/CS301R/301r_retnet/data/tokenizers/my_cool_tokenizer \
--dataset_name wikitext \
--seq_len 256 \
--dataset_dir /home/searlejj/fsl_groups/grp_retnet/compute/data \
--dataset_subset wikitext-2-v1 \
--text_feature text \
--splits 0.7 0.2 0.1 \
--rand_seed 42 \