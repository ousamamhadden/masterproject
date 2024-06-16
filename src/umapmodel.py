import logging
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
import pandas as pd
from happytransformer import HappyTextClassification, TCTrainArgs
from transformers import DistilBertConfig,DistilBertModel
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
    pipeline,
    FeatureExtractionPipeline
    )

os.environ["WANDB_PROJECT"] = "mlops-proj47"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def set_seed(seed: int):
    """Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` a
    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def usemodel(happy_tc,x):
    result = happy_tc.classify_text(x)
    #print(x)
    #print(type(result))  # <class 'happytransformer.happy_text_classification.TextClassificationResult'>
    #print(result)  # TextClassificationResult(label='POSITIVE', score=0.9998761415481567)
    #print(result.label)  # LABEL_1
    return result.label


@hydra.main(config_path="../config", config_name="default_config.yaml", version_base=None)
def train(config: DictConfig) -> None:
    """Train the model using the provided configuration."""
    model_name_path = 'models/model'

    config = AutoConfig.from_pretrained(model_name_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_path, config=config).distilbert
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name_path)
    #pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer,return_all_scores = True)
    mypipeline = pipeline(model=model, tokenizer=tokenizer, task="feature-extraction")

    df = pd.read_csv('data/processed/testdata.csv', sep=',')
    df = df.sample(frac=0.25, random_state=42)
    ypred = []
    ytest = []
    for index,row in df.iterrows():
        a = mypipeline(row['text'],return_tensors=True)
        b = a[0].numpy().mean(axis=0)
        print(b.shape)
        ypred.append(b)
        ytest.append(int(row['label']))
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(ypred)
    X_tsne = np.array(embedding) 
    y = np.array(ytest)
    plt.figure(figsize=(8, 6))
    colours = ListedColormap(['r','b','g','k'])

    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=colours)
    plt.legend(title="classes", *scatter.legend_elements())
    plt.show()
    plt.savefig('foobar.png')
    #plt.savefig('bar.png')

    #print(len(ypred[0][0]))
    print("DONE")

    

if __name__ == "__main__":
    train()
