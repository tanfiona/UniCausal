# You may call the pretrained models to test on your own datasets directly.

# Only caveat is they need to be processed into our CSV dataset format. 
# We created a helper function "_datasets/add_dummy_columns.py" to help. 
# But it is not perfect, especially for columns with different naming.

# In this file, we demonstrate how to "test" directly on an external corpus.
# We use the Causal News Corpus, a shared task of CASE 2022 @ EMNLP.
# Data source: https://github.com/tanfiona/CausalNewsCorpus/tree/master/data

### Subtask 1 : Sequence Classification ### 

sudo /home/fiona/anaconda3/envs/torchgeom/bin/python3 \
_datasets/add_dummy_columns.py --input_csv_path data/test_subtask1_text.csv

sudo CUDA_VISIBLE_DEVICES=1 /home/fiona/anaconda3/envs/torchgeom/bin/python3 \
run_seqbase.py \
--seq_val_file data/test_subtask1_text_formatted.csv --do_predict \
--model_name_or_path tanfiona/unicausal-seq-baseline \
--output_dir outs/causalnewscorpus_subtask1 \
--per_device_eval_batch_size 32 --seed 42

### Subtask 2 : Cause-Effect Span Detection ###
# Note that the original task also identifies "Signal" spans, but we ignore them for now.

sudo /home/fiona/anaconda3/envs/torchgeom/bin/python3 \
_datasets/add_dummy_columns.py --input_csv_path data/test_subtask2_text.csv

sudo CUDA_VISIBLE_DEVICES=1 /home/fiona/anaconda3/envs/torchgeom/bin/python3 \
run_tokbase.py \
--span_val_file data/test_subtask2_text_formatted.csv --do_predict \
--model_name_or_path tanfiona/unicausal-tok-baseline \
--output_dir outs/causalnewscorpus_subtask2 \
--per_device_eval_batch_size 32 --seed 42