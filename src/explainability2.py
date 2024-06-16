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
from transformers_interpret import MultiLabelClassificationExplainer

import numpy as np
import pandas as pd
from happytransformer import HappyTextClassification, TCTrainArgs
from transformers import DistilBertConfig,DistilBertModel
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
import matplotlib.pyplot as plt
os.environ["WANDB_PROJECT"] = "mlops-proj47"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@hydra.main(config_path="../config", config_name="default_config.yaml", version_base=None)
def train(config: DictConfig) -> None:
    model_name_path = 'models/model2label'

    config = AutoConfig.from_pretrained(model_name_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name_path)
    df = pd.read_csv('data/processed/traindata2labels.csv', sep=',')
    cls_explainer = MultiLabelClassificationExplainer(model, tokenizer)
    filegood = open('data/explainability/good2.txt', 'a')
    filebad = open('data/explainability/bad2.txt', 'a')
    for i in range(15000,25000):
        a = cls_explainer(df.iloc[i]['text'])
        atrribution = sorted(a['LABEL_0'], key= lambda x: x[1])
        badwords = [ x[0] for x in atrribution[:10]]
        for word in badwords:
            filebad.write(word + ",")
        filebad.write('\n')

        goodwords = [ x[0] for x in atrribution[-10:]]
        for word in goodwords:
            filegood.write(word + ",")
        filegood.write('\n')

    filegood.close()
    filebad.close()




    


    

if __name__ == "__main__":
    train()
