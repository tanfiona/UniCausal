#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
Source: https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner_no_trainer.py
"""

import argparse
import logging
import math
import os

import numpy as np
import random
from pathlib import Path
from datetime import datetime

import datasets
import torch
from datasets import load_metric
from _datasets.unifiedcre import load_cre_dataset, available_datasets
from _datasets.data_collator import DataCollatorForTokenClassification
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    # set_seed,
)
from transformers.file_utils import get_full_repo_name
from utils.files import set_seeds, make_dir
from utils.logger import get_logger
import torch.nn.functional as F


# logger = logging.getLogger(__name__)

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        nargs="*",         # 0 or more values expected => creates a list
        type=str,
        default=None,
        choices=available_datasets,
        help='List of datasets to include in run. Dataset must be available under "data/splits" folder and presplitted.'
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help= "Where do you want to store the pretrained models downloaded from huggingface.co"
    )
    parser.add_argument(
        "--pair_train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--pair_val_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ratio",
        nargs=3,         # pspan:apair:aseq ratio
        type=str,
        default=None,
        help='Define ratio to use, overwrites natural distribution.'
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk", "ce"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--do_train_val",
        action="store_true",
        help="Assign datasets and loaders by splits, do note to specify both datasets.",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Do training.",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Do validation.",
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Get predictions on validation/test set.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.pair_train_file is None and args.pair_val_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.pair_train_file is not None:
            extension = args.pair_train_file.split(".")[-1]
            assert extension in ["csv"], "`pair_train_file` should be a CSV file."
        if args.pair_val_file is not None:
            extension = args.pair_val_file.split(".")[-1]
            assert extension in ["csv"], "`pair_val_file` should be a CSV file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


PADDING_DICT = {
    'input_ids': 0,
    'tokens': '[PAD]',
    'attention_mask': 0,
    'labels': -100,
    'label': -100,
    'ce_tags': -100,
    'ce_tags1': -100,
    'ce_tags2': -100,
    'token_type_ids': 0
    }


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Set items
    set_seeds(args.seed)
    # Setup logging
    log_file_name = datetime.now().strftime('logfile_%Y_%m_%d_%H_%M_%S.log')
    log_save_path = f"{args.output_dir}/{log_file_name.lower()}"
    make_dir(log_save_path)
    logger = get_logger(log_save_path, no_stdout=False)
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.DEBUG if args.debug else (
            logging.INFO if accelerator.is_local_main_process else logging.ERROR)
            )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    seq_files = {}
    if args.pair_train_file is not None and args.do_train:
        seq_files["train"] = args.pair_train_file
    if args.pair_val_file is not None and (args.do_eval or args.do_predict):
        seq_files["validation"] = args.pair_val_file

    if args.dataset_name is not None:
        # Loading dataset from a predefined list and format.
        span_datasets, seq_datasets, stats = load_cre_dataset(
            args.dataset_name, args.do_train_val, 
            also_add_span_sequence_into_seq=True,
            seq_files=seq_files,
            do_train=args.do_train
            )
    else:
        # These do not work given current changes
        raise NotImplementedError

    # Trim a number of training examples
    if args.debug:
        for split in seq_datasets.keys():
            seq_datasets[split] = seq_datasets[split].select(range(100))
        print(seq_datasets)

    # Some hard coded portions
    text_column_name = "text"
    seq_label_column_name = "label"
    num_labels=2

    seq_structure_source = list(seq_datasets.keys())[0]
    logging.info('Using auto detected data source for column and data structures: '+\
        f'"{seq_structure_source}"')
 
    # Load pretrained model and tokenizer
    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    if not tokenizer_name_or_path:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir
    )
    # only needed for pair
    num_added_toks = tokenizer.add_tokens(['<ARG0>','</ARG0>','<ARG1>','</ARG1>'], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_seq_length" if args.pad_to_max_length else False

    # Tokenize all texts.
    def tokenize_and_add_tags(examples):

        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_seq_length,
            padding=padding,
            truncation=True,
            is_split_into_words=False
        )

        tokenized_inputs['labels'] = examples[seq_label_column_name]

        return tokenized_inputs

    with accelerator.main_process_first():
        processed_seq_datasets = seq_datasets.map(
            tokenize_and_add_tags,
            batched=True,
            remove_columns=seq_datasets[seq_structure_source].column_names,
            desc="Running tokenizer on dataset",
        )

    if args.do_train:
        # Log a few random samples from the training set:
        tmp_dataset = processed_seq_datasets["pair_train"]
        for index in random.sample(range(len(tmp_dataset)), 3):
            logger.info(f"Sample {index} of the PAIR training set: {tmp_dataset[index]}.")
            eg = seq_datasets["pair_train"][index]
            logger.info(f"Original form: {eg}.")

    if args.do_eval:  
        tmp_dataset = processed_seq_datasets["pair_validation"]
        for index in random.sample(range(len(tmp_dataset)), 3):
            logger.info(f"Sample {index} of the PAIR validation set: {tmp_dataset[index]}.")
            eg = seq_datasets["pair_validation"][index]
            logger.info(f"Original form: {eg}.")

    for k,v in seq_datasets.items():
        logger.info(f"{k}, n={len(v)}")
    

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )


    if args.do_train:
        train_dataset = processed_seq_datasets['pair_train']
        apair_dataloader = DataLoader(
            train_dataset, 
            shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size,
            drop_last=False
        )
    else:
        apair_dataloader = None
    
    if args.do_eval or args.do_predict: # To allow for predict from another test set
        eval_apair_dataloader = DataLoader(
            processed_seq_datasets['pair_validation'], 
            shuffle=False, collate_fn=data_collator, 
            batch_size=args.per_device_eval_batch_size
        )
        eval_apair_corpus_col = seq_datasets["pair_validation"]["corpus"]
        eval_apair_unique_corpus = list(set(eval_apair_corpus_col))
    else:
        eval_apair_dataloader = None


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, apair_dataloader, eval_apair_dataloader = accelerator.prepare(
            model, optimizer, apair_dataloader, eval_apair_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below 
    # (cause its length will be shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    if args.do_train:
        num_update_steps_per_epoch = math.ceil(args.per_device_train_batch_size / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

    # Train & Evaluate
    if args.do_train:
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

    if args.do_train or args.do_eval:

        for epoch in range(args.num_train_epochs):

            if args.do_train:
                model.train()
                for step, apair_batch in enumerate(apair_dataloader):

                    outputs = model(**apair_batch)

                    loss = outputs.loss
                    loss = loss / args.gradient_accumulation_steps
                    accelerator.backward(loss)

                    if step % args.gradient_accumulation_steps == 0 or step == len(apair_dataloader) - 1:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1

                    if completed_steps >= args.max_train_steps:
                        break

            if args.do_eval:
                
                pair_metric = {d:load_metric('utils/seq_metrics.py') for d in eval_apair_unique_corpus+['all']}

                model.eval()
                for step, batch in enumerate(tqdm(eval_apair_dataloader)):
                    with torch.no_grad():
                        outputs = model(**batch)

                    # Get Seq Predictions & References
                    seq_preds = outputs.logits.argmax(dim=-1).detach().cpu().clone().tolist()
                    seq_refs = batch["labels"].detach().cpu().clone().tolist()

                    # Add to metrics
                    pair_metric['all'].add_batch(
                        predictions=seq_preds,
                        references=seq_refs
                    )

                    # Add to metrics by dataset name
                    corps = eval_apair_corpus_col[step*args.per_device_eval_batch_size:(step+1)*args.per_device_eval_batch_size]
                    for i,d in enumerate(corps):
                        pair_metric[d].add(
                            prediction=seq_preds[i],
                            reference=seq_refs[i],
                        )
                
                # Print predictions
                for d in eval_apair_unique_corpus+['all']:
                    eval_metric = pair_metric[d].compute()
                    logger.info(f"epoch {epoch}: pair '{d}' : {eval_metric}")
                
                if not args.do_train:
                    # one round of evaluation is enough if we are not in training mode
                    break

            if args.do_train and args.push_to_hub and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    repo.push_to_hub(
                        commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                    )

    if args.do_train and args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)


    if args.do_predict:

        # reset in case do_eval was also evoked
        pair_metric = {d:load_metric('utils/seq_metrics.py') for d in eval_apair_unique_corpus+['all']}

        # set into eval mode
        model.eval()

        logger.info("***** Running predicting *****")
        logger.info(f"  Num pair examples = {len(eval_apair_corpus_col)}")
        all_pair_preds, all_pair_refs = [], []

        for step, batch in enumerate(tqdm(eval_apair_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)

            # Get Seq Predictions & References
            seq_preds = outputs.logits.argmax(dim=-1).detach().cpu().clone().tolist()
            seq_refs = batch["labels"].detach().cpu().clone().tolist()

            # Add to metrics
            pair_metric['all'].add_batch(
                predictions=seq_preds,
                references=seq_refs
            )

            # Add to metrics by dataset name
            corps = eval_apair_corpus_col[step*args.per_device_eval_batch_size:(step+1)*args.per_device_eval_batch_size]
            for i,d in enumerate(corps):
                pair_metric[d].add(
                    prediction=seq_preds[i],
                    reference=seq_refs[i],
                )
            
            # Store predictions
            all_pair_preds.extend(seq_preds)
            all_pair_refs.extend(seq_refs)


        # Print predictions
        for d in eval_apair_unique_corpus+['all']:
            eval_metric = pair_metric[d].compute()
            logger.info(f"pair predictions for '{d}' : {eval_metric}")

        # Save Predictions
        eval_apair_index_col = seq_datasets["pair_validation"]["index"]
        assert(len(all_pair_preds)==len(all_pair_refs))
        assert(len(eval_apair_index_col)==len(all_pair_preds))
        with open(os.path.join(args.output_dir,'pair_predictions.txt'), "w") as writer:
            writer.write("index\tpair_pred\tpair_label\n")
            for ix, (p,r) in enumerate(zip(all_pair_preds,all_pair_refs)):
                writer.write(f"{eval_apair_index_col[ix]}\t{p}\t{r}\n")

    logger.info('complete')


if __name__ == "__main__":
    main()
