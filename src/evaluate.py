import logging
import os
import random

import hydra
import numpy as np
import torch
from happytransformer import HappyTextClassification, TTTrainArgs
from omegaconf import DictConfig
from transformers import AutoTokenizer, DistilBertForSequenceClassification,DistilBertConfig,DistilBertModel
import pandas as pd

from happytransformer import HappyTextClassification, TCTrainArgs
def usemodel(happy_tc,x):
    result = happy_tc.classify_text(x)
    #print(type(result))  # <class 'happytransformer.happy_text_classification.TextClassificationResult'>
    print(result)  # TextClassificationResult(label='POSITIVE', score=0.9998761415481567)
    #print(result.label)  # LABEL_1
    return result.label

@hydra.main(config_path="../config", config_name="default_config.yaml", version_base=None)
def train(config: DictConfig) -> None:


    happy_tc = HappyTextClassification("DISTILBERT", "./models/model", num_labels=4)
    df = pd.read_csv('data/processed/traindata.csv', sep=',')
    df = df.sample(frac=0.015, random_state=42)

    ypred = []
    ytest = []
    for index,row in df.iterrows():
        ypred.append(usemodel(happy_tc,row['text']))
        ytest.append(int(row['label']))
    ypred = list(map(lambda x: int(x[-1]),ypred))
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(ytest,ypred))
    print(confusion_matrix(ytest,ypred))



if __name__ == "__main__":
    train()
