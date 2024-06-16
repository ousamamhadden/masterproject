from dataclasses import dataclass, field
import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoConfig, Trainer, EvalPrediction, set_seed
from transformers.training_args import TrainingArguments

import sys

from multimodal_transformers.data import load_data_from_folder
from multimodal_transformers.model import TabularConfig
from multimodal_transformers.model import AutoModelWithTabular
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO)
os.environ["COMET_MODE"] = "DISABLED"
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )


@dataclass
class MultimodalDataTrainingArguments:
    """
    Arguments pertaining to how we combine tabular features
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_path: str = field(
        metadata={"help": "the path to the csv file containing the dataset"}
    )
    column_info_path: str = field(
        default=None,
        metadata={
            "help": "the path to the json file detailing which columns are text, categorical, numerical, and the label"
        },
    )

    column_info: dict = field(
        default=None,
        metadata={
            "help": "a dict referencing the text, categorical, numerical, and label columns"
            "its keys are text_cols, num_cols, cat_cols, and label_col"
        },
    )

    categorical_encode_type: str = field(
        default="ohe",
        metadata={
            "help": "sklearn encoder to use for categorical data",
            "choices": ["ohe", "binary", "label", "none"],
        },
    )
    numerical_transformer_method: str = field(
        default="yeo_johnson",
        metadata={
            "help": "sklearn numerical transformer to preprocess numerical data",
            "choices": ["yeo_johnson", "box_cox", "quantile_normal", "none"],
        },
    )
    task: str = field(
        default="classification",
        metadata={
            "help": "The downstream training task",
            "choices": ["classification", "regression"],
        },
    )

    mlp_division: int = field(
        default=4,
        metadata={
            "help": "the ratio of the number of "
            "hidden dims in a current layer to the next MLP layer"
        },
    )
    combine_feat_method: str = field(
        default="individual_mlps_on_cat_and_numerical_feats_then_concat",
        metadata={
            "help": "method to combine categorical and numerical features, "
            "see README for all the method"
        },
    )
    mlp_dropout: float = field(
        default=0.1, metadata={"help": "dropout ratio used for MLP layers"}
    )
    numerical_bn: bool = field(
        default=True,
        metadata={"help": "whether to use batchnorm on numerical features"},
    )
    categorical_bn: bool = field(
        default=True,
        metadata={"help": "whether to use batchnorm on categorical features"},
    )
    use_simple_classifier: str = field(
        default=True,
        metadata={"help": "whether to use single layer or MLP as final classifier"},
    )
    mlp_act: str = field(
        default="relu",
        metadata={
            "help": "the activation function to use for finetuning layers",
            "choices": ["relu", "prelu", "sigmoid", "tanh", "linear"],
        },
    )
    gating_beta: float = field(
        default=0.2,
        metadata={
            "help": "the beta hyperparameters used for gating tabular data "
            "see https://www.aclweb.org/anthology/2020.acl-main.214.pdf"
        },
    )

    def __post_init__(self):
        assert self.column_info != self.column_info_path
        if self.column_info is None and self.column_info_path:
            with open(self.column_info_path, "r") as f:
                self.column_info = json.load(f)

# ,label,Price,Cleaning Fee,Availability 365,Host Response Rate,City

text_cols = ["text"]
cat_cols = ["City"]

numerical_cols = ["Price", "Cleaning Fee", "Availability 365","Host Response Rate"]

column_info_dict = {
    "text_cols": text_cols,
    "num_cols": numerical_cols,
    "cat_cols": cat_cols,
    "label_col": "label",
    "label_list": [0,1,2,3],
}


model_args = ModelArguments(model_name_or_path="mymodel")

data_args = MultimodalDataTrainingArguments(
    data_path="./../",
    #combine_feat_method="attention_on_cat_and_numerical_feats",
    combine_feat_method="individual_mlps_on_cat_and_numerical_feats_then_concat",
    column_info=column_info_dict,
    categorical_encode_type=None,
    task="classification",
)

training_args = TrainingArguments(
    output_dir="./logs/model_name",
    logging_dir="./logs/runs",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    logging_steps=25,
    eval_steps=250,
)

set_seed(training_args.seed)

tokenizer_path_or_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
print('Specified tokenizer: ', tokenizer_path_or_name)
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path_or_name,
    cache_dir=model_args.cache_dir,
)

# Get Datasets
train_dataset, val_dataset, test_dataset = load_data_from_folder(
    data_args.data_path,
    data_args.column_info["text_cols"],
    tokenizer,
    label_col=data_args.column_info["label_col"],
    label_list=data_args.column_info["label_list"],
    categorical_cols=None,
    numerical_cols=data_args.column_info["num_cols"],
    categorical_encode_type=None,
    sep_text_token_str=tokenizer.sep_token,
)
num_labels = len(np.unique(train_dataset.labels))

config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
)
tabular_config = TabularConfig(
    num_labels=num_labels,
    cat_feat_dim=0,
    numerical_feat_dim=train_dataset.numerical_feats.shape[1],
    **vars(data_args)
)
config.tabular_config = tabular_config
model = AutoModelWithTabular.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    config=config,
    cache_dir=model_args.cache_dir,
)

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
)


def calc_classification_metrics(p: EvalPrediction):
    predictions = p.predictions[0]
    pred_labels = np.argmax(predictions, axis=1)
    pred_scores = softmax(predictions, axis=1)[:, 1]
    labels = p.label_ids
    print(classification_report(labels,pred_labels))

print("Beginning training")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=calc_classification_metrics,
)
trainer.train()
print("FINISHED traning")
print("mlp")

trainer.evaluate(eval_dataset=test_dataset)
print("FINISHED")
