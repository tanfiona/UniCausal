CUDA_VISIBLE_DEVICES=0-7 python run.py --model_name_or_path bert-base-cased --dataset_name altlex because \
--output_dir outs/trial_run --num_train_epochs 20 --per_device_train_batch_size 128 --per_device_eval_batch_size 32 \
 --do_train_val --do_eval --do_predict --do_train