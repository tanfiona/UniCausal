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
from datasets import ClassLabel, load_metric, load_dataset
from _datasets.unifiedcre import load_cre_dataset, available_datasets
from _datasets.data_collator import DataCollatorForTokenClassification
from torch.utils.data import DataLoader, WeightedRandomSampler
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
    SchedulerType,
    default_data_collator,
    get_scheduler,
    # set_seed,
)
from transformers.file_utils import get_full_repo_name
from models.classifiers.modeling_bert import BertForUnifiedCRBase
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
        "--seq_train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--seq_val_file", type=str, default=None, help="A csv or a json file containing the validation data."
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
        "--alpha",
        type=int,
        default=3,
        help=(
            "Weight to focus on seq clf task compared to tok clf x 3"
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
    parser.add_argument(
        "--span_augment",
        action="store_true",
        help="Create negative examples from positive span examples.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.span_train_file is None and args.span_val_file is None \
        and args.seq_train_file is None and args.seq_val_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.span_train_file is not None:
            extension = args.span_train_file.split(".")[-1]
            assert extension in ["csv"], "`span_train_file` should be a CSV file."
        if args.span_val_file is not None:
            extension = args.span_val_file.split(".")[-1]
            assert extension in ["csv"], "`span_val_file` should be a CSV file."
        if args.seq_train_file is not None:
            extension = args.seq_train_file.split(".")[-1]
            assert extension in ["csv"], "`seq_train_file` should be a CSV file."
        if args.seq_val_file is not None:
            extension = args.seq_val_file.split(".")[-1]
            assert extension in ["csv"], "`seq_val_file` should be a CSV file."

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


def merge_dicts(d1, d2, d3):
    # combine batches in dim=0, pad at end
    # return {k:torch.cat([d1[k],d2[k]],dim=0) for k in d1.keys()}
    
    dummy = list(d1.keys())[0]
    d1_length = d1[dummy].shape[1]
    d2_length = d2[dummy].shape[1]
    d3_length = d3[dummy].shape[1]
    max_seq_length = max(d1_length, d2_length, d3_length)
    
    output_dict = {}
    for k in d1.keys():
        if k=='label':
            output_dict[k] = torch.cat((d1[k],d2[k],d3[k]),dim=0)
        else:
            output_dict[k] = torch.cat([
                F.pad(d1[k],(0,max_seq_length-d1_length),"constant",PADDING_DICT[k]),
                F.pad(d2[k],(0,max_seq_length-d2_length),"constant",PADDING_DICT[k]),
                F.pad(d3[k],(0,max_seq_length-d3_length),"constant",PADDING_DICT[k])
                ],dim=0)
    
    return output_dict


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def flatten(t):
    return [item for sublist in t for item in sublist]


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

    span_files, seq_files = {}, {}
    if args.span_train_file is not None and args.do_train:
        span_files["train"] = args.span_train_file
    if args.span_val_file is not None and (args.do_eval or args.do_predict):
        span_files["validation"] = args.span_val_file
    if args.seq_train_file is not None and args.do_train:
        seq_files["train"] = args.seq_train_file
    if args.seq_val_file is not None and (args.do_eval or args.do_predict):
        seq_files["validation"] = args.seq_val_file

    if args.dataset_name is not None:
        # Loading dataset from a predefined list and format.
        span_datasets, seq_datasets, stats = load_cre_dataset(
            args.dataset_name, args.do_train_val, 
            span_augment=args.span_augment,
            span_files=span_files, 
            seq_files=seq_files,
            do_train=args.do_train)
        min_batch_size, pspan, apair, aseq = stats
        logger.info(f"Minimum batch size needed: {min_batch_size} ;   pspan:apair:aseq ratio {pspan}:{apair}:{aseq}")
        if args.ratio is not None:
            pspan, apair, aseq = args.ratio
            pspan, apair, aseq = int(pspan), int(apair), int(aseq)
            min_batch_size = apair + aseq + pspan
            logger.info(f"OVERWRITE Minimum batch size needed: {min_batch_size} ;   pspan:apair:aseq ratio {pspan}:{apair}:{aseq}")
    else:
        # These do not work given current changes
        raise NotImplementedError
        # data_files = {}
        # if args.span_train_file is not None and args.do_train_val:
        #     data_files["train"] = args.span_train_file
        #     extension = args.span_train_file.split(".")[-1]
        # if args.span_val_file is not None:
        #     data_files["validation"] = args.span_val_file
        #     extension = args.span_val_file.split(".")[-1]
        # span_datasets = load_dataset(extension,data_files=data_files)

        # data_files = {}
        # if args.seq_train_file is not None and args.do_train_val:
        #     data_files["train"] = args.seq_train_file
        #     extension = args.seq_train_file.split(".")[-1]
        # if args.seq_val_file is not None:
        #     data_files["validation"] = args.seq_val_file
        #     extension = args.seq_val_file.split(".")[-1]
        # seq_datasets = load_dataset(extension,data_files=data_files)
    
        # # Segment the datasets
        # if args.span_train_file is not None and args.seq_train_file is not None and args.do_train_val:
        #     # positive_spans == span_train
        #     posi_spans_ids = span_datasets["train"]["id"]
        #     neg_seqs_ids, posi_seqs_ids = [], []
        #     for ix, d in enumerate(seq_datasets["train"]):
        #         if int(d["label"])==0:
        #             neg_seqs_ids.append(ix)
        #         elif d["index"] in posi_spans_ids:
        #             posi_seqs_ids.append(ix)
            
        #     lowest_denom = min(len(posi_spans_ids), len(neg_seqs_ids), len(posi_seqs_ids))
        #     pspan = int(len(posi_spans_ids)/lowest_denom)
        #     pseq = int(len(posi_seqs_ids)/lowest_denom)
        #     nseq = int(len(neg_seqs_ids)/lowest_denom)
        #     min_batch_size = pspan + pseq + nseq
        #     seq_datasets["posi_train"] = seq_datasets["train"].select(posi_seqs_ids)
        #     seq_datasets["neg_train"] = seq_datasets["train"].select(neg_seqs_ids)
        #     del(seq_datasets["train"])
        #     logger.info(f"Minimum batch size needed: {min_batch_size} ;   pspan:pseq:nseq ratio {pspan}:{pseq}:{nseq}")

    # Trim a number of training examples
    if args.debug:
        for s in span_datasets.keys():
            span_datasets[s] = span_datasets[s].select(range(min(100,len(span_datasets[S]))))
        print(span_datasets)
        for s in seq_datasets.keys():
            seq_datasets[s] = seq_datasets[s].select(range(min(100,len(seq_datasets[s]))))
        print(seq_datasets)

    # Some hard coded portions
    text_column_name = "text"
    span_label_column_name = "ce_tags"
    seq_label_column_name = "label"

    span_structure_source = list(span_datasets.keys())[0]
    seq_structure_source = list(seq_datasets.keys())[0]
    logging.info('Using auto detected data source for column and data structures: '+\
        f'"{span_structure_source}" and "{seq_structure_source}"')
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
        label_to_id = {'B-C': 0, 'B-E': 1, 'I-C': 2, 'I-E': 3, '_': 4}
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
    num_added_toks = tokenizer.add_tokens(['<ARG0>','</ARG0>','<ARG1>','</ARG1>'], special_tokens=True)

    if args.model_name_or_path:
        model = BertForUnifiedCRBase.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            num_seq_labels=2,
            loss_function='simple',
            alpha=args.alpha
        )
    else:
        logger.info("Training new model from scratch")
        model = BertForUnifiedCRBase.from_config(config)

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

        tags, tags1, tags2 = [], [], []
        for i, (label,label1,label2) in enumerate(zip(\
            examples[span_label_column_name],
            examples[f"{span_label_column_name}1"],
            examples[f"{span_label_column_name}2"],
            )):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids, label_ids1, label_ids2 = [], [], []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(PADDING_DICT[span_label_column_name])
                    label_ids1.append(PADDING_DICT[f"{span_label_column_name}1"])
                    label_ids2.append(PADDING_DICT[f"{span_label_column_name}2"])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    label_ids1.append(label_to_id[label1[word_idx]])
                    label_ids2.append(label_to_id[label2[word_idx]])
                # For the other tokens in a word, we set the label to either the 
                # current label or -100, depending on the label_all_tokens flag.
                else:
                    if args.label_all_tokens:
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                        label_ids1.append(b_to_i_label[label_to_id[label1[word_idx]]])
                        label_ids2.append(b_to_i_label[label_to_id[label2[word_idx]]])
                    else:
                        label_ids.append(PADDING_DICT[span_label_column_name])
                        label_ids1.append(PADDING_DICT[f"{span_label_column_name}1"])
                        label_ids2.append(PADDING_DICT[f"{span_label_column_name}2"])
                previous_word_idx = word_idx
            tags.append(label_ids)
            tags1.append(label_ids1)
            tags2.append(label_ids2)
        
        tokenized_inputs[span_label_column_name] = tags
        tokenized_inputs[f"{span_label_column_name}1"] = tags1
        tokenized_inputs[f"{span_label_column_name}2"] = tags2
        tokenized_inputs[seq_label_column_name] = examples[seq_label_column_name]
    
        return tokenized_inputs


    def tokenize_and_add_tags(examples):

        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_seq_length,
            padding=padding,
            truncation=True,
            is_split_into_words=False
        )

        dummy_span_labels = [] # missing spans / we don't want to train or evaluate on them
        for ids in tokenized_inputs["input_ids"]: # list of list
            sequence_length = len(ids)
            dummy_span_labels.append([PADDING_DICT[span_label_column_name]]*sequence_length)
        
        tokenized_inputs[span_label_column_name] = dummy_span_labels
        tokenized_inputs[f"{span_label_column_name}1"] = dummy_span_labels
        tokenized_inputs[f"{span_label_column_name}2"] = dummy_span_labels
        tokenized_inputs[seq_label_column_name] = examples[seq_label_column_name]

        return tokenized_inputs

    
    with accelerator.main_process_first():
        processed_span_datasets = span_datasets.map(
            tokenize_and_align_tags,
            batched=True,
            remove_columns=span_datasets[span_structure_source].column_names,
            desc="Running tokenizer on dataset",
        )
        processed_seq_datasets = seq_datasets.map(
            tokenize_and_add_tags,
            batched=True,
            remove_columns=seq_datasets[seq_structure_source].column_names,
            desc="Running tokenizer on dataset",
        )

    if args.do_train:
        # Log a few random samples from the training set:
        tmp_dataset = processed_span_datasets["span_train"]
        for index in random.sample(range(len(tmp_dataset)), 3):
            logger.info(f"Sample {index} of the SPAN training set: {tmp_dataset[index]}.")
            eg = span_datasets["span_train"][index]
            logger.info(f"Original form: {eg}.")
        
        tmp_dataset = processed_seq_datasets["seq_train"]
        for index in random.sample(range(len(tmp_dataset)), 3):
            logger.info(f"Sample {index} of the SEQ training set: {tmp_dataset[index]}.")
            eg = seq_datasets["seq_train"][index]
            logger.info(f"Original form: {eg}.")

        tmp_dataset = processed_seq_datasets["pair_train"]
        for index in random.sample(range(len(tmp_dataset)), 3):
            logger.info(f"Sample {index} of the PAIR training set: {tmp_dataset[index]}.")
            eg = seq_datasets["pair_train"][index]
            logger.info(f"Original form: {eg}.")

    if args.do_eval:
        print(processed_span_datasets)
        print(processed_seq_datasets)
        tmp_dataset = processed_span_datasets["span_validation"]
        for index in random.sample(range(len(tmp_dataset)), 2):
            logger.info(f"Sample {index} of the SPAN validation set: {tmp_dataset[index]}.")
            eg = span_datasets["span_validation"][index]
            logger.info(f"Original form: {eg}.")

    for k,v in span_datasets.items():
        logger.info(f"{k}, n={len(v)}")
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


    def get_sampler(dataset, batch_size):

        y = dataset[seq_label_column_name]
        class_count=dict(Counter(y))
        class_count=np.array(list(class_count.values()))
        weight=1./class_count
        samples_weight = np.array([weight[t] for t in y])
        samples_weight=torch.from_numpy(samples_weight)

        sampler = WeightedRandomSampler(
            weights=samples_weight, 
            num_samples=batch_size,
            replacement=True
            )
        logger.info(f'Prepared sampler: {sampler}')
        return sampler


    if args.do_train:
        multiples = int(args.per_device_train_batch_size/min_batch_size)
        train_dataset = processed_span_datasets['span_train']
        pspan_dataloader = DataLoader(
            train_dataset, 
            shuffle=True, collate_fn=data_collator, batch_size=multiples*pspan,
            drop_last=False
        )
        aseq_dataloader = DataLoader(
            processed_seq_datasets['seq_train'], 
            sampler=get_sampler(processed_seq_datasets['seq_train'], multiples*aseq),
            collate_fn=data_collator, batch_size=multiples*aseq
        )
        apair_dataloader = DataLoader(
            processed_seq_datasets['pair_train'], 
            sampler=get_sampler(processed_seq_datasets['pair_train'], multiples*apair),
            collate_fn=data_collator, batch_size=multiples*apair
        )
    else:
        pspan_dataloader, aseq_dataloader, apair_dataloader = \
            None, None, None
    
    if args.do_eval or args.do_predict: # To allow for predict from another test set
        eval_dataset = processed_span_datasets['span_validation']
        eval_pspan_dataloader = DataLoader(
            eval_dataset, 
            shuffle=False, collate_fn=data_collator, 
            batch_size=args.per_device_eval_batch_size
            )
        eval_pspan_corpus_col = span_datasets["span_validation"]["corpus"]
        eval_pspan_unique_corpus = list(set(eval_pspan_corpus_col))

        eval_aseq_dataloader = DataLoader(
            processed_seq_datasets['seq_validation'], 
            shuffle=False, collate_fn=data_collator, 
            batch_size=args.per_device_eval_batch_size
        )
        eval_aseq_corpus_col = seq_datasets["seq_validation"]["corpus"]
        eval_aseq_unique_corpus = list(set(eval_aseq_corpus_col+eval_pspan_unique_corpus))

        eval_apair_dataloader = DataLoader(
            processed_seq_datasets['pair_validation'], 
            shuffle=False, collate_fn=data_collator, 
            batch_size=args.per_device_eval_batch_size
        )
        eval_apair_corpus_col = seq_datasets["pair_validation"]["corpus"]
        eval_apair_unique_corpus = list(set(eval_apair_corpus_col))
    else:
        eval_pspan_dataloader, eval_aseq_dataloader, eval_apair_dataloader =\
            None, None, None

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
    model, optimizer, pspan_dataloader, aseq_dataloader, apair_dataloader, \
        eval_pspan_dataloader, eval_aseq_dataloader, eval_apair_dataloader = accelerator.prepare(
            model, optimizer, pspan_dataloader, aseq_dataloader, apair_dataloader, \
                eval_pspan_dataloader, eval_aseq_dataloader, eval_apair_dataloader
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
        # Remove if all are "_" (dummy examples)
        true_predictions, true_labels = [], []
        for pred, gold_label in zip(y_pred, y_true):
            true_p, true_l = [], []
            for (p, l) in zip(pred, gold_label):
                if l != ignore_ids:
                    true_p.append(label_list[p])
                    true_l.append(label_list[l])
            if len(set(true_l))==1 and true_l[0]=='_' and remove_if_no_ce: # all dummy values
                # drop these examples, append empties for alignment to index
                true_predictions.append([])
                true_labels.append([])
            else:
                true_predictions.append(true_p)
                true_labels.append(true_l)

        return true_predictions, true_labels


    def compute_metrics(d='all'):
        try:
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
        except:
            # No C-E true labels available
            return {
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "accuracy": 0,
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
        logger.info(f"  Distributed batch size across pspan:apair:aseq = {multiples*pspan}:{multiples*apair}:{multiples*aseq}")
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
                # for step, batch in enumerate(train_dataloader):
                for step, (pspan_batch, aseq_batch, apair_batch) in enumerate(\
                    zip(pspan_dataloader, cycle(aseq_dataloader), cycle(apair_dataloader))):

                    logging.debug('pspan input_ids.shape:'+str(pspan_batch["input_ids"].shape))
                    logging.debug('aseq input_ids.shape:'+str(aseq_batch["input_ids"].shape))
                    logging.debug('apair input_ids.shape:'+str(apair_batch["input_ids"].shape))

                    pspan_bs = pspan_batch["input_ids"].shape[0]
                    aseq_bs = aseq_batch["input_ids"].shape[0]
                    apair_bs = apair_batch["input_ids"].shape[0]

                    # Get Mask for Span Examples
                    span_egs_mask = [1]*pspan_bs+[0]*aseq_bs+[0]*apair_bs

                    # Insert to model
                    outputs = model( 
                        span_egs_mask=torch.LongTensor(span_egs_mask).to(device),
                        **merge_dicts(pspan_batch, aseq_batch, apair_batch)
                        )
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
                seq_metric = {d:load_metric('utils/seq_metrics.py') for d in eval_aseq_unique_corpus+['all']}
                pair_metric = {d:load_metric('utils/seq_metrics.py') for d in eval_apair_unique_corpus+['all']} 

                model.eval()
                for step, batch in enumerate(tqdm(eval_pspan_dataloader)):
                    with torch.no_grad():
                        outputs = model(**batch)

                    # Get Span Predictions & References
                    preds, refs = format(
                        predictions=outputs.tok_logits.argmax(dim=-1), 
                        labels=batch[span_label_column_name]
                        )
                    preds1, refs1 = format(
                        predictions=outputs.tok_logits.argmax(dim=-1), 
                        labels=batch[f"{span_label_column_name}1"]
                        )
                    preds2, refs2 = format(
                        predictions=outputs.tok_logits.argmax(dim=-1), 
                        labels=batch[f"{span_label_column_name}2"]
                        )
                    
                    # Get Seq Predictions & References
                    seq_preds = outputs.seq_logits.argmax(dim=-1).detach().cpu().clone().tolist()
                    seq_refs = batch[seq_label_column_name].detach().cpu().clone().tolist()

                    # Add to metrics
                    metric['all'].add_batch(
                        predictions=preds,
                        references=refs 
                    ) # predictions and preferences are expected to be a nested list of labels, not label_ids
                    metric['all'].add_batch(
                        predictions=preds1,
                        references=refs1 
                    )
                    metric['all'].add_batch(
                        predictions=preds2,
                        references=refs2 
                    )
                    seq_metric['all'].add_batch(
                        predictions=seq_preds,
                        references=seq_refs
                    )

                    # Add to metrics by dataset name
                    corps = eval_pspan_corpus_col[step*args.per_device_eval_batch_size:(step+1)*args.per_device_eval_batch_size]
                    for i,d in enumerate(corps):
                        metric[d].add(
                            prediction=preds[i],
                            reference=refs[i],
                        )
                        metric[d].add(
                            prediction=preds1[i],
                            reference=refs1[i],
                        )
                        metric[d].add(
                            prediction=preds2[i],
                            reference=refs2[i],
                        )
                        seq_metric[d].add(
                            prediction=seq_preds[i],
                            reference=seq_refs[i],
                        )

                for step, batch in enumerate(tqdm(eval_aseq_dataloader)):
                    with torch.no_grad():
                        outputs = model(**batch)

                    # Get Seq Predictions & References
                    seq_preds = outputs.seq_logits.argmax(dim=-1).detach().cpu().clone().tolist()
                    seq_refs = batch[seq_label_column_name].detach().cpu().clone().tolist()

                    # Add to metrics
                    seq_metric['all'].add_batch(
                        predictions=seq_preds,
                        references=seq_refs
                    )

                    # Add to metrics by dataset name
                    corps = eval_aseq_corpus_col[step*args.per_device_eval_batch_size:(step+1)*args.per_device_eval_batch_size]
                    for i,d in enumerate(corps):
                        seq_metric[d].add(
                            prediction=seq_preds[i],
                            reference=seq_refs[i],
                        )

                for step, batch in enumerate(tqdm(eval_apair_dataloader)):
                    with torch.no_grad():
                        outputs = model(**batch)

                    # Get Seq Predictions & References
                    seq_preds = outputs.seq_logits.argmax(dim=-1).detach().cpu().clone().tolist()
                    seq_refs = batch[seq_label_column_name].detach().cpu().clone().tolist()

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
                for d in eval_pspan_unique_corpus+['all']:
                    eval_metric = compute_metrics(d)
                    logger.info(f"epoch {epoch}: span '{d}' : {eval_metric}")
                for d in list(set(eval_pspan_unique_corpus+eval_aseq_unique_corpus))+['all']:
                    eval_metric = seq_metric[d].compute()
                    logger.info(f"epoch {epoch}: seq '{d}' : {eval_metric}")
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
        metric = {d:load_metric('seqeval') for d in eval_pspan_unique_corpus+['all']}
        seq_metric = {d:load_metric('utils/seq_metrics.py') for d in eval_aseq_unique_corpus+['all']}
        pair_metric = {d:load_metric('utils/seq_metrics.py') for d in eval_apair_unique_corpus+['all']} 

        # set into eval mode
        model.eval()

        logger.info("***** Running predicting *****")
        logger.info(f"  Num span examples = {len(eval_pspan_corpus_col)}")
        logger.info(f"  Num seq examples = {len(eval_aseq_corpus_col)}")
        logger.info(f"  Num pair examples = {len(eval_apair_corpus_col)}")

        all_preds, all_refs = [], []
        all_seq_preds, all_seq_refs = [], []
        all_pair_preds, all_pair_refs = [], []

        for step, batch in enumerate(tqdm(eval_pspan_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)

            # Get Span Predictions & References
            preds, refs = format(
                predictions=outputs.tok_logits.argmax(dim=-1), 
                labels=batch[span_label_column_name],
                remove_if_no_ce=False
                )
            preds1, refs1 = format(
                predictions=outputs.tok_logits.argmax(dim=-1), 
                labels=batch[f"{span_label_column_name}1"],
                remove_if_no_ce=False
                )
            preds2, refs2 = format(
                predictions=outputs.tok_logits.argmax(dim=-1), 
                labels=batch[f"{span_label_column_name}2"],
                remove_if_no_ce=False
                )
            
            # Get Seq Predictions & References
            seq_preds = outputs.seq_logits.argmax(dim=-1).detach().cpu().clone().tolist()
            seq_refs = batch[seq_label_column_name].detach().cpu().clone().tolist()
            
            # Add to metrics
            metric['all'].add_batch(
                predictions=preds,
                references=refs 
            ) # predictions and preferences are expected to be a nested list of labels, not label_ids
            metric['all'].add_batch(
                predictions=preds1,
                references=refs1 
            )
            metric['all'].add_batch(
                predictions=preds2,
                references=refs2 
            )
            seq_metric['all'].add_batch(
                predictions=seq_preds,
                references=seq_refs
            )

            # Add to metrics by dataset name
            corps = eval_pspan_corpus_col[step*args.per_device_eval_batch_size:(step+1)*args.per_device_eval_batch_size]
            for i,d in enumerate(corps):
                metric[d].add(
                    prediction=preds[i],
                    reference=refs[i],
                )
                metric[d].add(
                    prediction=preds1[i],
                    reference=refs1[i],
                )
                metric[d].add(
                    prediction=preds2[i],
                    reference=refs2[i],
                )
                seq_metric[d].add(
                    prediction=seq_preds[i],
                    reference=seq_refs[i],
                )

            # Store predictions
            all_preds.extend(preds)
            all_refs.extend(refs)
            all_preds.extend(preds1)
            all_refs.extend(refs1)
            all_preds.extend(preds2)
            all_refs.extend(refs2)
            all_seq_preds.extend(seq_preds)
            all_seq_refs.extend(seq_refs)

        for step, batch in enumerate(tqdm(eval_aseq_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)

            # Get Seq Predictions & References
            seq_preds = outputs.seq_logits.argmax(dim=-1).detach().cpu().clone().tolist()
            seq_refs = batch[seq_label_column_name].detach().cpu().clone().tolist()

            # Add to metrics
            seq_metric['all'].add_batch(
                predictions=seq_preds,
                references=seq_refs
            )

            # Add to metrics by dataset name
            corps = eval_aseq_corpus_col[step*args.per_device_eval_batch_size:(step+1)*args.per_device_eval_batch_size]
            for i,d in enumerate(corps):
                seq_metric[d].add(
                    prediction=seq_preds[i],
                    reference=seq_refs[i],
                )
            
            # Store predictions
            all_seq_preds.extend(seq_preds)
            all_seq_refs.extend(seq_refs)

        for step, batch in enumerate(tqdm(eval_apair_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)

            # Get Seq Predictions & References
            seq_preds = outputs.seq_logits.argmax(dim=-1).detach().cpu().clone().tolist()
            seq_refs = batch[seq_label_column_name].detach().cpu().clone().tolist()

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
        for d in eval_pspan_unique_corpus+['all']:
            eval_metric = compute_metrics(d)
            logger.info(f"span predictions for '{d}' : {eval_metric}")
        for d in list(set(eval_pspan_unique_corpus+eval_aseq_unique_corpus))+['all']:
            eval_metric = seq_metric[d].compute()
            logger.info(f"seq predictions for '{d}' : {eval_metric}")
        for d in eval_apair_unique_corpus+['all']:
            eval_metric = pair_metric[d].compute()
            logger.info(f"pair predictions for '{d}' : {eval_metric}")

        # Save Predictions
        eval_pspan_index_col = span_datasets["span_validation"]["index"]
        eval_aseq_index_col = eval_pspan_index_col+seq_datasets["seq_validation"]["index"]
        eval_apair_index_col = seq_datasets["pair_validation"]["index"]
        eval_pspan_index_col = list(eval_pspan_index_col)*3

        assert(len(all_preds)==len(all_refs))
        assert(len(eval_pspan_index_col)==len(all_preds))
        with open(os.path.join(args.output_dir,'span_predictions.txt'), "w") as writer:
            writer.write("index\tpred\tce_tags\n")
            for ix, (p,r) in enumerate(zip(all_preds,all_refs)):
                writer.write(f"{eval_pspan_index_col[ix]}\t{p}\t{r}\n")

        assert(len(all_seq_preds)==len(all_seq_refs))
        assert(len(eval_aseq_index_col)==len(all_seq_preds))
        with open(os.path.join(args.output_dir,'seq_predictions.txt'), "w") as writer:
            writer.write("index\tseq_pred\tseq_label\n")
            for ix, (p,r) in enumerate(zip(all_seq_preds,all_seq_refs)):
                writer.write(f"{eval_aseq_index_col[ix]}\t{p}\t{r}\n")
        
        assert(len(all_pair_preds)==len(all_pair_refs))
        assert(len(eval_apair_index_col)==len(all_pair_preds))
        with open(os.path.join(args.output_dir,'pair_predictions.txt'), "w") as writer:
            writer.write("index\tpair_pred\tpair_label\n")
            for ix, (p,r) in enumerate(zip(all_pair_preds,all_pair_refs)):
                writer.write(f"{eval_apair_index_col[ix]}\t{p}\t{r}\n")
    
    logger.info('complete')


if __name__ == "__main__":
    main()
