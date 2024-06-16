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

os.environ["WANDB_PROJECT"] = "Master-Thesis123"
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
    cfg = config.training
    model = HappyTextClassification("DISTILBERT", "distilbert-base-uncased", num_labels=4)
    model.model.config.max_position_embeddings = 2048
    configuration = DistilBertConfig(max_position_embeddings=2048)

    distilbert = DistilBertModel(config=configuration).to("cuda:0")
    model.model.distilbert = distilbert
    '''
    if cfg.metric_tracker != "wandb":  # necessary for unit testing
        args = TCTrainArgs(batch_size=cfg.batch_size, learning_rate=cfg.lr, num_train_epochs=cfg.epochs,eval_ratio=0.05)
    else:
        args = TCTrainArgs(
            batch_size=cfg.batch_size, learning_rate=cfg.lr, num_train_epochs=cfg.epochs,eval_ratio=0.05
        )'''
    args = TCTrainArgs(num_train_epochs=3, batch_size=32, eval_ratio=0.05, report_to="wandb" )
    model.train('data/processed/traindata.csv', args=args)
    #set_seed(cfg.seed)
    logging.info("Training model...")
    #model.train(cfg.dataset_path, args=args)
    logging.info("Training complete.")
    #model.save('models/model/')
    logging.info("Model saved to %s", cfg.model_path)

    df = pd.read_csv('data/processed/testdata.csv', sep=',')
    ypred = []
    ytest = []
    for index,row in df.iterrows():
        ypred.append(usemodel(model,row['text']))
        ytest.append(int(row['label']))
    ypred = list(map(lambda x: int(x[-1]),ypred))
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(ytest,ypred))
    print(confusion_matrix(ytest,ypred))

    


    

if __name__ == "__main__":
    train()
