#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from sklearn.preprocessing import MinMaxScaler

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import EarlyStoppingCallback, IntervalStrategy

from model import MarianMTModel
from model_config import MarianConfigCustom

#TODO custom torch dataset
from data_loader import TranslationDataset
from attributor.NeuralModule import NeuralModule

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.24.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    num_heads_attr: int = field( default=6, metadata={"help": "The number of heads that attributes will be added to the model."})

    operator : str = field(default=None, metadata={"help": "The operator to use for the attribute addition. Choose from add, replace, multiply, average."})
    enc_att_op : bool = field(default=False, metadata={"help":"if ENCODER attention is to be used with attribution."})
    cross_att_op : bool = field(default=False, metadata={"help":"if CROSS attention is to be used with attribution."})
    scaler : str = field(default=None, metadata={"help": "The operator to scale the attributions. Choose minmax, softmax, or one_hot."})
    injection_pretransform : str = field(default=None, metadata={"help": "If the pretransformation should be applied to the attributions before injecting them"})
    approx_attributions_model : str = field(default=None, metadata={"help": "Specify the model to use to generate approximate attributions instead of the dataset"})
    freeze_approx_model : bool = field(default=False, metadata={"help": "Whether to freeze the approx_attributions_model or not"})
    reinit_approx_model : bool = field(default=False, metadata={"help": "Whether to reinitialize the approx_attributions_model weights"})
    generated : bool = field(default=False, metadata={"help": "to use generated target sequences for training."})
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})
    target_attr_external: str = field(default=None, metadata={"help": "Target attribute from another method."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacrebleu) on a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the metrics (sacrebleu) on a jsonlines file."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the :obj:`decoder_start_token_id`.Useful for"
                " multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token needs to"
                " be the target language token.(Usually it is the target language token)"
            )
        },
    )
    metrics: Optional[str] = field(
        default="sacrebleu",
        metadata={
            "help": (
                "Comma-separated metrics to compute. Supported: sacrebleu, chrf, ter, rouge. "
                "Example: --metrics sacrebleu,chrf"
            )
        },
    )
    

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        elif self.source_lang is None or self.target_lang is None:
            raise ValueError("Need to specify the source language and the target language.")

        # accepting both json and jsonl file extensions, as
        # many jsonlines files actually have a .json extension
        valid_extensions = ["json", "jsonl"]

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in valid_extensions, "`train_file` should be a jsonlines file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in valid_extensions, "`validation_file` should be a jsonlines file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    source_lang = data_args.source_lang
    target_lang = data_args.target_lang


    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_translation", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is expected, e.g. with "
            "`--source_prefix 'translate English to German: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    print("modelarge:",model_args) 
    if data_args.dataset_name is not None and data_args.dataset_name != "custom" and "attribution" not in data_args.dataset_name and data_args.dataset_name != "custom100k":
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.dataset_name == "custom":
        #load the custom data set
        data_files = {"train": f'data/{source_lang}-{target_lang}/train_filtered_full.json', "test": f'data/{source_lang}-{target_lang}/test_filtered_full.json', "validation": f'data/{source_lang}-{target_lang}/validation_filtered_full.json'}#train_filtered_baseline2
        raw_datasets = load_dataset("json", data_files=data_files, cache_dir='./data_cache')
        
    elif data_args.dataset_name == "custom100k":
        if model_args.model_name_or_path.startswith("Helsinki"):
            data_files = {"train": f'attributions/{source_lang}-{target_lang}/Marian/without_attr/train/*.json', "test": f'attributions/{source_lang}-{target_lang}/Marian/without_attr/test/*.json', "validation": f'attributions/{source_lang}-{target_lang}/Marian/without_attr/validation/*.json'}#train_filtered_baseline2
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir='./data_cache')
        elif model_args.model_name_or_path.startswith("facebook"):
            data_files = {"train": f'attributions/{source_lang[:2]}-{target_lang[:2]}/MBart/without_attr/train/*.json', "test": f'attributions/{source_lang[:2]}-{target_lang[:2]}/MBart/without_attr/test/*.json', "validation": f'attributions/{source_lang[:2]}-{target_lang[:2]}/MBart/without_attr/validation/*.json'}#train_filtered_baseline2
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir='./data_cache')
        else:
            data_files = {"train": f'attributions/{source_lang}-{target_lang}/200_k/train/*.json', "test": f'attributions/{source_lang}-{target_lang}/200_k/test/*.json', "validation": f'attributions/{source_lang}-{target_lang}/200_k/validation/*.json'}#train_filtered_baseline2
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir='./data_cache')
    elif "attribution" in data_args.dataset_name:
        xi_name = data_args.dataset_name.split('_')[1:]
        xi_name = '_'.join(xi_name)
        # if data_args.target_attr_external:
            # data_files = {"train": f'attributions/{source_lang}-{target_lang}/{xi_name}/train/*.json', "test": f'attributions/{source_lang}-{target_lang}/{data_args.target_attr_external}/test/*.json', "validation": f'attributions/{source_lang}-{target_lang}/{xi_name}/validation/*.json'}#train_filtered_baseline2
        
        def load_and_fix(pattern, col_name="attribution"):
            ds = load_dataset("json", data_files={"x": pattern}, split="x")

            # Unify column name if some files use "attributions"
            if col_name not in ds.column_names and "attributions" in ds.column_names:
                ds = ds.rename_column("attributions", col_name)

            # Make sure every row is a list of floats
            def to_float_list(ex):
                v = ex[col_name]
                if v is None:
                    return {col_name: []}
                if isinstance(v, list):
                    return {col_name: [float(x) for x in v]}
                # wrap scalars
                return {col_name: [float(v)]}
            ds = ds.map(to_float_list, desc=f"Normalize {col_name} to list<float>")

            # Now cast using HF Features (not pyarrow)
            ds = ds.cast_column(col_name, Sequence(Value("float32")))
            return ds

        if model_args.model_name_or_path.startswith("facebook"):
            train = load_and_fix(f'attributions/{source_lang[:2]}-{target_lang[:2]}/MBart/{xi_name}/train/*.json')
            valid = load_and_fix(f'attributions/{source_lang[:2]}-{target_lang[:2]}/MBart/{xi_name}/validation/*.json')
            if data_args.target_attr_external:
                test  = load_and_fix(f'attributions/{source_lang}-{target_lang}/{data_args.target_attr_external}/test/*.json')
            else:
                test  = load_and_fix(f'attributions/{source_lang[:2]}-{target_lang[:2]}/MBart/{xi_name}/test/*.json')


        else:
            if model_args.generated:
                train = load_and_fix(f'attributions/{source_lang}-{target_lang}/Marian/{xi_name}/generated/train/*.json')
                valid = load_and_fix(f'attributions/{source_lang}-{target_lang}/Marian/{xi_name}/generated/validation/*.json')
            else:
                train = load_and_fix(f'attributions/{source_lang}-{target_lang}/Marian/{xi_name}/train/*.json')
                valid = load_and_fix(f'attributions/{source_lang}-{target_lang}/Marian/{xi_name}/validation/*.json')
            if data_args.target_attr_external:
                test  = load_and_fix(f'attributions/{source_lang}-{target_lang}/{data_args.target_attr_external}/test/*.json')
            else:
                if model_args.generated:
                    test  = load_and_fix(f'attributions/{source_lang}-{target_lang}/Marian/{xi_name}/generated/test/*.json')
                else:
                    test  = load_and_fix(f'attributions/{source_lang}-{target_lang}/Marian/{xi_name}/test/*.json')

        # Make columns consistent across splits (drop extras if needed)
        common = set(train.column_names) & set(valid.column_names) & set(test.column_names)
        train = train.remove_columns([c for c in train.column_names if c not in common])
        valid = valid.remove_columns([c for c in valid.column_names if c not in common])
        test  = test.remove_columns([c for c in test.column_names if c not in common])
        
        raw_datasets = DatasetDict(train=train, validation=valid, test=test)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        num_heads_attr=model_args.num_heads_attr,
        operator=model_args.operator,
        encoder_attribution_op = model_args.enc_att_op,
        cross_attribution_op = model_args.cross_att_op,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    default_state_dict = model.state_dict()

    if model_args.model_name_or_path == "facebook/mbart-large-50":
        custom_config = MarianConfigCustom.from_json_file('checkpoints/initial_checkpoint_custom/config_mbart.json')
    elif source_lang == "de":
        custom_config = MarianConfigCustom.from_json_file('checkpoints/initial_checkpoint_custom/config_de_en.json')
    elif source_lang == "ar":
        custom_config = MarianConfigCustom.from_json_file('checkpoints/initial_checkpoint_custom/config_ar_en.json')
    elif source_lang == "fr":
        custom_config = MarianConfigCustom.from_json_file('checkpoints/initial_checkpoint_custom/config_fr_en.json')
    elif source_lang == "en" and target_lang == "zh":
        custom_config = MarianConfigCustom.from_json_file('checkpoints/initial_checkpoint_custom/config_en_zh.json')
    elif source_lang == "es" and target_lang == "it":
        custom_config = MarianConfigCustom.from_json_file('checkpoints/initial_checkpoint_custom/config_es_it.json')
    elif source_lang == "en" and target_lang == "da":
        custom_config = MarianConfigCustom.from_json_file('checkpoints/initial_checkpoint_custom/config_en_da.json')
    else:
        raise ValueError(f"Source language {source_lang} not supported. Please use de, ar, fr, en_zh or es_it.")

    custom_config.num_heads_attr = model_args.num_heads_attr
    custom_config.operator = model_args.operator
    custom_config.encoder_attribution_op = model_args.enc_att_op
    custom_config.cross_attribution_op = model_args.cross_att_op
    custom_config.approx_attributions_model = model_args.approx_attributions_model
    custom_config.injection_pretransform = model_args.injection_pretransform
    custom_config.freeze_approx_model = model_args.freeze_approx_model
    
    custom_config.vocab_size = config.vocab_size

    AutoConfig.register("custom_marian", MarianConfigCustom)
    AutoModelForSeq2SeqLM.register(MarianConfigCustom, MarianMTModel)

    custom_model = AutoModelForSeq2SeqLM.from_config(custom_config)

    if not training_args.do_train:
        custom_model = custom_model.from_pretrained(f'{training_args.output_dir}', config=custom_config)
    model = custom_model
    logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.resize_token_embeddings(len(tokenizer))

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    source_lang = data_args.source_lang.split("_")[0]
    target_lang = data_args.target_lang.split("_")[0]
    if training_args.do_train:
        #column_names = raw_datasets["train"].column_names
        column_names = [target_lang, source_lang]
    elif training_args.do_eval:
        #column_names = raw_datasets["validation"].column_names
        column_names = [target_lang, source_lang]
    elif training_args.do_predict:
        #column_names = raw_datasets["test"].column_names
        column_names = [target_lang, source_lang]
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return
    
    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert data_args.target_lang is not None and data_args.source_lang is not None, (
            f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --source_lang and "
            "--target_lang arguments."
        )

        tokenizer.src_lang = data_args.source_lang
        tokenizer.tgt_lang = data_args.target_lang
        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id


    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length"# if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def _scaler_minmax(attributions):
        scaler = MinMaxScaler()
        tensor_np = attributions
        scaled_np = scaler.fit_transform(tensor_np)
        return scaled_np
    
    def _softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # subtract max for numerical stability
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def _softmax_to_one_hot(matrix):
        softmax_matrix = _softmax(matrix)
        max_indices = np.argmax(softmax_matrix, axis=1)
        one_hot_matrix = np.zeros_like(matrix)
        one_hot_matrix[np.arange(matrix.shape[0]), max_indices] = 1
        return one_hot_matrix
    
    def almost_diagonal_with_shifts(rows, cols, band_width=1.0,
                                shift_prob=0.2, max_total_shift=3,
                                seed=None):
        """
        Create a rectangular matrix with a diagonal band that sometimes shifts right.

        rows, cols        : shape of the matrix
        band_width        : how wide the band around the diagonal is
        shift_prob        : probability of shifting the band right at each row
        max_total_shift   : max cumulative shift to the right (in columns)
        seed              : random seed for reproducibility
        """
        rng = np.random.default_rng(seed)

        # row and column indices
        i, j = np.indices((rows, cols))  # i: row index, j: col index

        # "ideal" diagonal line from top-left to bottom-right (scaled to rectangular shape)
        row_idx = np.arange(rows)
        base_center = row_idx * (cols - 1) / max(rows - 1, 1)   # shape (rows,)

        # build a piecewise-constant shift that occasionally steps to the right
        shift = np.zeros(rows)
        for r in range(1, rows):
            shift[r] = shift[r - 1]
            if rng.random() < shift_prob and shift[r] < max_total_shift:
                shift[r] += 1  # step one column to the right at this row and onward

        # center column for each row, including shifts
        center = base_center + shift  # shape (rows,)

        # distance of each cell from the shifted "diagonal" at its row
        dist = j - center[:, None]

        # Gaussian band around the diagonal
        M = np.exp(-(dist**2) / (2 * band_width**2))

        return M

    def preprocess_function(examples):
        """Channge translation to targetString ["sourceString"]"""
        inputs = [ex for ex in examples[source_lang]]
        targets = [ex for ex in examples[target_lang]]
        
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        if  "attribution" in data_args.dataset_name:
            
            attributions = []
            if 'shape_src' in examples:

                for samples, shapes in zip(examples['attribution'], examples['shape_src']):
                    
                    reshaped_tensor = np.array(samples).reshape(int(shapes[0]), int(shapes[1]))
                    
                    if model_args.scaler == "minmax":
                        reshaped_tensor = _scaler_minmax(reshaped_tensor)
                    elif model_args.scaler == "one_hot":
                        reshaped_tensor = _softmax_to_one_hot(reshaped_tensor)
                    elif model_args.scaler == "random":
                        reshaped_tensor = np.random.rand(int(shapes[0]), int(shapes[1]))
                    elif model_args.scaler == "one_diagonal":
                        reshaped_tensor = almost_diagonal_with_shifts(int(shapes[0]), int(shapes[1]), band_width=1.0,
                                shift_prob=0.2, max_total_shift=3, seed=None)   
                    elif model_args.scaler and model_args.scaler not in ["minmax", "one_hot", "softmax", "random", "one_diagonal"]:
                        raise ValueError(f"Scaler {model_args.scaler} not recognized. Choose from minmax, one_hot, or softmax.")

                    zero_tensor = np.zeros((data_args.max_source_length, data_args.max_source_length), dtype=np.float16)
                    # if we use approx_attribution_model, we fill the rest of the matrix with uniform distribution
                    # so that the sum of each row is 1
                    if model_args.approx_attributions_model is not None:
                        for i in range(shapes[1], data_args.max_source_length):
                            zero_tensor[i] = 1.0 / data_args.max_source_length
                        zero_tensor = zero_tensor.transpose(-1,-2)

                    zero_tensor[0:shapes[0], 0:shapes[1]] = reshaped_tensor
                    zero_tensor = zero_tensor.astype(np.float16)
                    attributions.append(zero_tensor)
                    
            else:
                
                for samples in examples['attribution']:
                    reshaped_tensor = np.array(samples).reshape(128, 128)  # Assuming the shape is (128, 128) for each attribution sample
                    if model_args.scaler == "minmax":
                        reshaped_tensor = _scaler_minmax(reshaped_tensor)
                    elif model_args.scaler == "one_hot":
                        reshaped_tensor = _softmax_to_one_hot(reshaped_tensor)
                    attributions.append(reshaped_tensor)

            model_inputs["attributions"] = np.stack(attributions, axis=0)
        # Tokenize targets with the `text_target` keyword argument
            model_inputs["attributions"] = np.moveaxis(model_inputs["attributions"],-1,-2)  

        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        if 'shape_src' in examples:
            model_inputs['shape_src'] = examples['shape_src']
        model_inputs["labels"] = labels["input_ids"]
            
        return model_inputs


    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

        # max_train_samples = min(len(train_dataset), 200_000)  # Use 200,000 samples at most
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
            
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            


    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]

        max_eval_samples = min(len(eval_dataset), 5000)  # Use 10,000 samples at most
        eval_dataset = eval_dataset.select(range(max_eval_samples))
        
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        # Keep an unprocessed copy of the original texts for saving alongside predictions
        orig_predict_dataset = predict_dataset
        
        # max_eval_samples = min(len(predict_dataset), 100)  # Use 1000 samples at most
        # predict_dataset = predict_dataset.select(range(max_eval_samples))
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
            # Apply the exact same selection to the original (unprocessed) dataset to keep alignment
            try:
                orig_predict_dataset = orig_predict_dataset.select(range(max_predict_samples))
            except Exception:
                pass
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metrics
    # Parse requested metrics list
    requested_metrics = set(m.strip().lower() for m in (data_args.metrics or "").split(",") if m.strip())
    if not requested_metrics:
        requested_metrics = {"sacrebleu"}

    supported_metrics = {"sacrebleu", "chrf", "ter", "rouge"}
    unknown = requested_metrics - supported_metrics
    if unknown:
        logger.warning(f"Ignoring unsupported metrics: {sorted(list(unknown))}")
        requested_metrics = requested_metrics & supported_metrics

    loaded_metrics = {}
    for m in requested_metrics:
        try:
            # rouge requires additional config to compute rougeLsum
            if m == "rouge":
                loaded_metrics[m] = evaluate.load("rouge", cache_dir=model_args.cache_dir, keep_in_memory=True)
            else:
                loaded_metrics[m] = evaluate.load(m, cache_dir=model_args.cache_dir, keep_in_memory=True)
        except Exception as e:
            logger.warning(f"Failed to load metric '{m}': {e}. It will be skipped.")


    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Post-process
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        # Prepare references format for sacrebleu/chrf/ter (list of list)
        references_list = [[lbl] for lbl in decoded_labels]

        results = {}
        if "sacrebleu" in loaded_metrics:
            try:
                sac = loaded_metrics["sacrebleu"].compute(predictions=decoded_preds, references=references_list)
                results["bleu"] = sac.get("score", sac.get("bleu", None))
            except Exception as e:
                logger.warning(f"sacrebleu failed: {e}")
        if "chrf" in loaded_metrics:
            try:
                ch = loaded_metrics["chrf"].compute(predictions=decoded_preds, references=references_list)
                # evaluate's chrf returns {'score': x, 'char_order': ..., 'word_order': ...}
                results["chrf"] = ch.get("score")
            except Exception as e:
                logger.warning(f"chrf failed: {e}")
        if "ter" in loaded_metrics:
            try:
                ter_res = loaded_metrics["ter"].compute(predictions=decoded_preds, references=references_list)
                # evaluate's ter returns {'score': x}
                results["ter"] = ter_res.get("score")
            except Exception as e:
                logger.warning(f"ter failed: {e}")
        if "rouge" in loaded_metrics:
            try:
                # ROUGE expects refs as list[str]
                rouge = loaded_metrics["rouge"].compute(
                    predictions=decoded_preds, references=decoded_labels, use_stemmer=True
                )
                # Keep a compact subset
                for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
                    if k in rouge:
                        results[k] = rouge[k]
            except Exception as e:
                logger.warning(f"rouge failed: {e}")

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        results["gen_len"] = float(np.mean(prediction_lens)) if prediction_lens else 0.0
        results = {k: round(v, 4) for k, v in results.items() if isinstance(v, (int, float))}
        return results

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,# if training_args.predict_with_generate else None,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset) + 1)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate: 
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))

                # Also save a pandas DataFrame with original source/target and generated text
                try:
                    # Reuse the same language keys as earlier
                    src_key = source_lang
                    tgt_key = target_lang
                    # Some datasets may not have both keys; handle gracefully
                    sources = orig_predict_dataset[src_key] if src_key in orig_predict_dataset.column_names else [""] * len(predictions)
                    references = orig_predict_dataset[tgt_key] if tgt_key in orig_predict_dataset.column_names else [""] * len(predictions)
                    idx = orig_predict_dataset["idx"] if "idx" in orig_predict_dataset.column_names else [""] * len(predictions)

                    # Truncate lists to the shortest length to avoid mismatch
                    n = min(len(predictions), len(sources), len(references))
                    data_rows = {
                        "source": sources[:n],
                        "reference": references[:n],
                        "generated": predictions[:n],
                        "idx": idx[:n],
                    }
                    df = pandas.DataFrame(data_rows)
                    output_prediction_df_file = os.path.join(training_args.output_dir, f"{training_args.run_name}_generated_predictions_with_refs.csv")
                    df.to_csv(output_prediction_df_file, index=False)
                    logger.info(f"Saved predictions with references to {output_prediction_df_file}")
                except Exception as e:
                    logger.warning(f"Failed to save predictions DataFrame with original texts: {e}")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "translation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    languages = [l for l in [data_args.source_lang, data_args.target_lang] if l is not None]
    if len(languages) > 0:
        kwargs["language"] = languages

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
