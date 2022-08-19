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
from collections import Counter
from datasets import ClassLabel, load_metric
from _datasets.unifiedcre import load_span_dataset_ungrouped, available_datasets
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
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
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
        "--span_train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--span_val_file", type=str, default=None, help="A csv or a json file containing the validation data."
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
    if args.dataset_name is None and args.span_train_file is None and args.span_val_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.span_train_file is not None:
            extension = args.span_train_file.split(".")[-1]
            assert extension in ["csv"], "`span_train_file` should be a CSV file."
        if args.span_val_file is not None:
            extension = args.span_val_file.split(".")[-1]
            assert extension in ["csv"], "`span_val_file` should be a CSV file."

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

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
    # 'tokens' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    span_files = {}
    if args.span_train_file is not None and args.do_train:
        span_files["train"] = args.span_train_file
    if args.span_val_file is not None and (args.do_eval or args.do_predict):
        span_files["validation"] = args.span_val_file

    # Loading dataset from a predefined list and format.
    span_datasets = load_span_dataset_ungrouped(
        args.dataset_name, args.do_train_val,
        span_files=span_files
        )
    
    # Trim a number of training examples
    if args.debug:
        for split in span_datasets.keys():
            span_datasets[split] = span_datasets[split].select(range(100))
        print(span_datasets)

    # Some hard coded portions
    text_column_name = "text"
    span_label_column_name = "ce_tags"

    if args.span_train_file is not None or args.dataset_name is not None:
        span_structure_source = 'span_train'
    elif args.span_val_file is not None:
        span_structure_source = 'span_validation'
    else:
        raise ValueError('We do not have any pair train or validation datasets to work with.')
    features = span_datasets[span_structure_source].features

    # In the event the labels are not a `Sequence[ClassLabel]`, 
    # we will need to go through the dataset to get the unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if args.do_train:
        if hasattr(features[span_label_column_name], 'feature') and isinstance(features[span_label_column_name].feature, ClassLabel):
            label_list = features[span_label_column_name].feature.names
            # No need to convert the labels since they are already ints.
            label_to_id = {i: i for i in range(len(label_list))}
        else:
            label_list = get_label_list(span_datasets[span_structure_source][span_label_column_name])
            label_to_id = {l: i for i, l in enumerate(label_list)}
    else: 
        # When not training, it is dangerous to let auto detection of label_to_id
        # the order can change, leading to wrong or missing predictions
        label_to_id = {'B-C': 0, 'B-E': 1, 'I-C': 2, 'I-E': 3, 'O': 4}
        label_list = list(label_to_id.keys())

    num_labels = len(label_list)
    logger.info(f"label_to_id: {label_to_id}")

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, num_labels=num_labels)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    if not tokenizer_name_or_path:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    # num_added_toks = tokenizer.add_tokens(['<ARG0>','</ARG0>','<ARG1>','</ARG1>'], special_tokens=True)

    if args.model_name_or_path:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForTokenClassification.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_seq_length" if args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_tags(examples):

        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_seq_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        tags = []
        for i, label in enumerate(examples[span_label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(PADDING_DICT[span_label_column_name])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the 
                # current label or -100, depending on the label_all_tokens flag.
                else:
                    if args.label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(PADDING_DICT[span_label_column_name])
                previous_word_idx = word_idx
            tags.append(label_ids)
        
        tokenized_inputs['labels'] = tags
    
        return tokenized_inputs

    
    with accelerator.main_process_first():
        processed_span_datasets = span_datasets.map(
            tokenize_and_align_tags,
            batched=True,
            remove_columns=span_datasets[span_structure_source].column_names,
            desc="Running tokenizer on dataset",
        )

    if args.do_train:
        # Log a few random samples from the training set:
        tmp_dataset = processed_span_datasets["span_train"]
        for index in random.sample(range(len(tmp_dataset)), 3):
            logger.info(f"Sample {index} of the SPAN training set: {tmp_dataset[index]}.")
            eg = span_datasets["span_train"][index]
            logger.info(f"Original form: {eg}.")

    if args.do_eval:  
        tmp_dataset = processed_span_datasets["span_validation"]
        for index in random.sample(range(len(tmp_dataset)), 2):
            logger.info(f"Sample {index} of the SPAN validation set: {tmp_dataset[index]}.")
            eg = span_datasets["span_validation"][index]
            logger.info(f"Original form: {eg}.")

    for k,v in span_datasets.items():
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
        train_dataset = processed_span_datasets['span_train']
        pspan_dataloader = DataLoader(
            train_dataset, 
            shuffle=True, collate_fn=data_collator, 
            batch_size=args.per_device_train_batch_size,
            drop_last=False
        )
    else:
        pspan_dataloader = None
    
    if args.do_eval or args.do_predict: # To allow for predict from another test set
        eval_dataset = processed_span_datasets['span_validation']
        eval_pspan_dataloader = DataLoader(
            eval_dataset,
            shuffle=False, collate_fn=data_collator, 
            batch_size=args.per_device_eval_batch_size
            )
        eval_pspan_corpus_col = span_datasets["span_validation"]["corpus"]
        eval_pspan_unique_corpus = list(set(eval_pspan_corpus_col))
    else:
        eval_pspan_dataloader = None

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
    model, optimizer, pspan_dataloader, eval_pspan_dataloader = accelerator.prepare(
        model, optimizer, pspan_dataloader, eval_pspan_dataloader
        )

    # Note -> the training dataloader needs to be prepared before we grab his length below 
    # (cause its length will be shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    if args.do_train:
        num_update_steps_per_epoch = math.ceil(len(pspan_dataloader) / args.gradient_accumulation_steps)
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

    def get_labels(predictions, references, ignore_ids=-100, remove_if_no_ce=True):
        # Transform predictions and references tensors to numpy arrays
        if device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        # Remove if all are 'O' (dummy examples)
        true_predictions, true_labels = [], []
        for pred, gold_label in zip(y_pred, y_true):
            true_p, true_l = [], []
            for (p, l) in zip(pred, gold_label):
                if l != ignore_ids:
                    true_p.append(label_list[p])
                    true_l.append(label_list[l])
            if len(set(true_l))==1 and true_l[0]=='O' and remove_if_no_ce: # all dummy values
                # drop these examples, append empties for alignment to index
                true_predictions.append([])
                true_labels.append([])
            else:
                true_predictions.append(true_p)
                true_labels.append(true_l)

        return true_predictions, true_labels


    def compute_metrics(d='all'):
        results = metric[d].compute()
        if args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # Train & Evaluate

    def format(predictions, labels, remove_if_no_ce=True):
        if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=PADDING_DICT[span_label_column_name])
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=PADDING_DICT[span_label_column_name])
        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)
        return get_labels(predictions_gathered, labels_gathered, PADDING_DICT[span_label_column_name], remove_if_no_ce)


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
                for step, batch in enumerate(pspan_dataloader):

                    outputs = model(**batch)
                    loss = outputs.loss
                    loss = loss / args.gradient_accumulation_steps
                    accelerator.backward(loss)
                    if step % args.gradient_accumulation_steps == 0 or step == len(pspan_dataloader) - 1:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1

                    if completed_steps >= args.max_train_steps:
                        break

            if args.do_eval:

                metric = {d:load_metric('seqeval') for d in eval_pspan_unique_corpus+['all']}

                model.eval()
                for step, batch in enumerate(tqdm(eval_pspan_dataloader)):
                    with torch.no_grad():
                        outputs = model(**batch)

                    # Get Span Predictions & References
                    preds, refs = format(
                        predictions=outputs.logits.argmax(dim=-1), 
                        labels=batch["labels"]
                        )

                    # Add to metrics
                    metric['all'].add_batch(
                        predictions=preds,
                        references=refs 
                    ) # predictions and preferences are expected to be a nested list of labels, not label_ids

                    # Add to metrics by dataset name
                    corps = eval_pspan_corpus_col[step*args.per_device_eval_batch_size:(step+1)*args.per_device_eval_batch_size]
                    for i,d in enumerate(corps):
                        metric[d].add(
                            prediction=preds[i],
                            reference=refs[i],
                        )

                # Print predictions
                for d in eval_pspan_unique_corpus+['all']:
                    eval_metric = compute_metrics(d)
                    logger.info(f"epoch {epoch}: span '{d}' : {eval_metric}")

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

        metric = {d:load_metric('seqeval') for d in eval_pspan_unique_corpus+['all']}

        # set into eval mode
        model.eval()

        logger.info("***** Running predicting *****")
        logger.info(f"  Num span examples = {len(eval_pspan_corpus_col)}")

        all_preds, all_refs = [], []

        for step, batch in enumerate(tqdm(eval_pspan_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)

            # Get Span Predictions & References
            preds, refs = format(
                predictions=outputs.logits.argmax(dim=-1), 
                labels=batch["labels"],
                remove_if_no_ce=False
                )
            
            # Add to metrics
            metric['all'].add_batch(
                predictions=preds,
                references=refs 
            ) # predictions and preferences are expected to be a nested list of labels, not label_ids

            # Add to metrics by dataset name
            corps = eval_pspan_corpus_col[step*args.per_device_eval_batch_size:(step+1)*args.per_device_eval_batch_size]
            for i,d in enumerate(corps):
                metric[d].add(
                    prediction=preds[i],
                    reference=refs[i],
                )

            # Store predictions
            all_preds.extend(preds)
            all_refs.extend(refs)

        # Print predictions
        for d in eval_pspan_unique_corpus+['all']:
            eval_metric = compute_metrics(d)
            logger.info(f"span predictions for '{d}' : {eval_metric}")

        # Save Predictions
        eval_pspan_index_col = span_datasets["span_validation"]["index"]
        assert(len(all_preds)==len(all_refs))
        assert(len(eval_pspan_index_col)==len(all_preds))
        with open(os.path.join(args.output_dir,'span_predictions.txt'), "w") as writer:
            writer.write("index\tpred\tce_tags\n")
            for ix, (p,r) in enumerate(zip(all_preds,all_refs)):
                writer.write(f"{eval_pspan_index_col[ix]}\t{p}\t{r}\n")

    logger.info('complete')


if __name__ == "__main__":
    main()
