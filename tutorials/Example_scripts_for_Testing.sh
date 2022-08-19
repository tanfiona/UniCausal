# You may call the pretrained models to test on your own datasets directly.
# Only caveat is they need to be processed into our CSV dataset format.
# In this file, we demonstrate how to "test" on the Causal News Corpus, a shared task of CASE 2022 @ EMNLP.
# Data source: https://github.com/tanfiona/CausalNewsCorpus/tree/master/data


### Subtask 2 ###

sudo /home/fiona/anaconda3/envs/torchgeom/bin/python3 \
_datasets/add_dummy_columns.py --input_csv_path data/test_subtask2_text.csv

sudo CUDA_VISIBLE_DEVICES=1 /home/fiona/anaconda3/envs/torchgeom/bin/python3 \
run_tokbase.py \
--suppress_test_eval \
--span_val_file data/test_subtask2_text_formatted.csv --do_predict \
--model_name_or_path tanfiona/unicausal-tok-baseline \
--output_dir outs/causalnewscorpus_subtask2 \
--per_device_eval_batch_size 32 --seed 42