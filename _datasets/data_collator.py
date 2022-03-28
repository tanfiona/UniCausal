from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.file_utils import PaddingStrategy
from transformers.models.bert import BertTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")


@dataclass
class DataCollatorForTokenClassification(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        label_name = "ce_tags" #"label" if "label" in features[0].keys() else "labels"
        list_of_label_names = [c for c in features[0].keys() if label_name in c] # "ce_tags1", "ce_tags2"
        
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if len(list_of_label_names)==0 else None,
        )

        if len(list_of_label_names)==0: #labels is None:
            return batch
        
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        for label_name in list_of_label_names:
            labels = [feature[label_name] for feature in features] #if label_name in features[0].keys() else None
            if padding_side == "right":
                batch[label_name] = [
                    list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
                ]
            else:
                batch[label_name] = [
                    [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
                ]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch

    def tf_call(self, features):
        import tensorflow as tf

        label_name = "ce_tags" #"label" if "label" in features[0].keys() else "labels"
        list_of_label_names = [c for c in features[0].keys() if label_name in c] # "ce_tags1", "ce_tags2"
        
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="tf" if len(list_of_label_names)==0 else None,
        )

        if len(list_of_label_names)==0: #labels is None:
            return batch

        sequence_length = tf.convert_to_tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        for label_name in list_of_label_names:
            labels = [feature[label_name] for feature in features] #if label_name in features[0].keys() else None
            if padding_side == "right":
                batch["labels"] = [
                    list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
                ]
            else:
                batch["labels"] = [
                    [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
                ]

        batch = {k: tf.convert_to_tensor(v, dtype=tf.int64) for k, v in batch.items()}
        return batch

    def numpy_call(self, features):
        import numpy as np

        label_name = "ce_tags" #"label" if "label" in features[0].keys() else "labels"
        list_of_label_names = [c for c in features[0].keys() if label_name in c] # "ce_tags1", "ce_tags2"

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="np" if len(list_of_label_names)==0 else None,
        )

        if len(list_of_label_names)==0: #labels is None:
            return batch

        sequence_length = np.array(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        for label_name in list_of_label_names:
            labels = [feature[label_name] for feature in features] #if label_name in features[0].keys() else None
            if padding_side == "right":
                batch["labels"] = [
                    list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
                ]
            else:
                batch["labels"] = [
                    [self.label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
                ]

        batch = {k: np.array(v, dtype=np.int64) for k, v in batch.items()}
        return batch
