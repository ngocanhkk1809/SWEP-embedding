import gc
import json
import wandb
import logging
import math
import os
import sys
import re
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Union, Any, Mapping
import torch
from torch import nn
from tqdm import tqdm
from datasets import Dataset, load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoModelForMaskedLM,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    AdamW,
    get_linear_schedule_with_warmup
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from models import VariationalModel
from data_utils import GetDataloader

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "Add dropout to the noise perturbation"},
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "Regularization term for KL loss"},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    only_base_model: bool = field(
        default=False,
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

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated. Default to the max input length of the model."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    train_batch_size: int = field(
        default=4,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    eval_batch_size: int = field(
        default=4,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def _prepare_input(data: Union[torch.Tensor, Any], device='cpu') -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}

        return data.to(**kwargs)
    return data


def _prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]], device) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    inputs = _prepare_input(inputs, device)
    if len(inputs) == 0:
        raise ValueError(
            "The batch received was empty, your model won't be able to train on it. Double-check that your "
            f"training dataset contains keys expected by the model."
        )

    return inputs


def save_model(args, model, epoch):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    ckpt_file = os.path.join(args.output_dir, f"bert_base_epoch_{epoch}.pt")
    ckpt = {"args": args, "state_dict": model.state_dict()}
    torch.save(ckpt, ckpt_file)


def processFile(file_text):
    sentences = re.split(r'\.\s*', file_text)
    sentences = [sentence.strip() for sentence in sentences if sentence]
    number_of_sentences = len(sentences)
    return sentences, number_of_sentences

def word_occur_in_sentences(sentences):
    word_occur_in_sentences = {}
    for sentence in sentences:
        sentence_words = sentence.lower().split()
        word_counts = Counter(sentence_words)
        word_occur_in_sentences[sentence] = word_counts
    return word_occur_in_sentences

def calculate_tfidf_weights(word_occur_in_sentences, number_of_sentences):
    tfidf_weights = {}
    for sentence, word_counts in word_occur_in_sentences.items():
        sentence_tfidf_weights = {}
        for word, count in word_counts.items():
            tf = count / len(sentence.split())
            num_sentences_with_word = sum([1 for sent in word_occur_in_sentences if word in word_occur_in_sentences[sent]])
            idf = math.log(number_of_sentences / (1 + num_sentences_with_word))
            tfidf_weight = tf * idf
            sentence_tfidf_weights[word] = tfidf_weight
        tfidf_weights[sentence] = sentence_tfidf_weights
    return tfidf_weights

def calculate_entropy(sentences):
    all_words = [word for sentence in sentences for word in sentence.lower().split()]
    word_counts = Counter(all_words)
    total_words = sum(word_counts.values())
    
    entropy_values = {}
    for word, count in word_counts.items():
        p_w = count / total_words
        if p_w > 0:
            entropy_values[word] = -p_w * math.log(p_w)
        else:
            entropy_values[word] = 0
    
    return entropy_values

def calculate_importance(sentences, tfidf_weights, entropy_values):
    importance_scores = []
    for sentence in sentences:
        words = sentence.lower().split()
        tfidf_scores = [tfidf_weights[sentence][word] if word in tfidf_weights[sentence] else 0 for word in words]
        entropy_scores = [entropy_values[word] if word in entropy_values else 0 for word in words]

        tfidf_norm = sum(tfidf_scores)
        entropy_norm = sum(entropy_scores)

        importance_scores.append([
            (tfidf / tfidf_norm if tfidf_norm > 0 else 0) + (entropy / entropy_norm if entropy_norm > 0 else 0)
            for tfidf, entropy in zip(tfidf_scores, entropy_scores)
        ])
    return importance_scores

def mask_high_importance_words(sentences, importance_scores, window_size):
    masked_sentences = []
    for sentence, scores in zip(sentences, importance_scores):
        words = sentence.split()
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        for i in sorted_indices[:window_size]:
            surrounding_words = []
            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if j != i:
                    surrounding_words.append(words[j])
            
            if surrounding_words:
                words[i] = surrounding_words[0]  
        
        masked_sentences.append(' '.join(words))
    return masked_sentences

def randomly_permute_remaining_words(sentences, max_percentage):
    permuted_sentences = []
    for sentence in sentences:
        words = sentence.split()
        num_to_permute = int(len(words) * max_percentage)
        indices_to_permute = random.sample(range(len(words)), num_to_permute)
        random.shuffle(indices_to_permute)
        for i, j in zip(sorted(indices_to_permute), indices_to_permute):
            words[i], words[j] = words[j], words[i]
        permuted_sentences.append(' '.join(words))
    return permuted_sentences

def custom_data_collator(features):
    sentences = [feature['text'] for feature in features]
    sentences, num_sentences = processFile('\n'.join(sentences))
    word_occur = word_occur_in_sentences(sentences)
    tfidf_weights = calculate_tfidf_weights(word_occur, num_sentences)
    entropy_values = calculate_entropy(sentences)
    importance_scores = calculate_importance(sentences, tfidf_weights, entropy_values)
    masked_sentences = mask_high_importance_words(sentences, importance_scores, window_size=5)
    permuted_sentences = randomly_permute_remaining_words(masked_sentences, max_percentage=0.1)
    
    for i, sentence in enumerate(permuted_sentences):
        features[i]['input_ids'] = tokenizer(sentence, truncation=True, padding='max_length')['input_ids']
    
    return features


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(extension, data_files=data_files)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    padding = "max_length" if data_args.pad_to_max_length else False

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return tokenizer(examples["text"], padding=padding, truncation=True, max_length=data_args.max_seq_length)

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=[text_column_name],
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Use the custom data collator
    data_collator = custom_data_collator

    device = torch.cuda.current_device()

    model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, config=config)

    model.resize_token_embeddings(len(tokenizer))

    model = VariationalModel(model, model_args, only_base_model=model_args.only_base_model)
    model = model.to(device)

    dataloader = GetDataloader(model,
                               tokenized_datasets["train"],
                               tokenized_datasets["validation"],
                               data_collator,
                               training_args)

    train_dataloader = dataloader.get_train_dataloader()

    # Training loop here (omitted for brevity)

if __name__ == "__main__":
    main()
