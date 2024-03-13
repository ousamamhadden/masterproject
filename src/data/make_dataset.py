import csv
import logging
import os

import pandas as pd
import hydra
from datasets import IterableDataset, load_dataset
from omegaconf import DictConfig
import math
import string
from sklearn.model_selection import train_test_split
# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
def replaceempty(x):
    if x == "":
        return "@"
    else:
        return x
def numtoClass(x):
    if x > 9:
        return 10
    else:
        return math.ceil(x)


def is_english_char(char):
    # Check if the character is a letter, digit, common punctuation, space, or newline in English
    return char.isascii() and (char.isalnum() or char in string.punctuation or char.isspace())

def filter_non_english(text):
    # Filter out non-English characters from the text
    english_chars = [char for char in text if is_english_char(char)]
    return ''.join(english_chars)

def cleancolumn(df, columnname):
    cleancolumnname = columnname + 'Clean'
    df[cleancolumnname] = df[columnname].fillna('@').map(lambda x : filter_non_english(x))
    df[cleancolumnname] = df[cleancolumnname].map(lambda x : replaceempty(x))


@hydra.main(config_path='../../config', config_name="default_config.yaml", version_base = None)
def make_dataset(cfg: DictConfig) -> None:
    """Downloads dataset from huggingface hub and generates a CSV file.

    Args:
        cfg (DictConfig): The configuration object.

    Returns:
        None

    Raises:
        ValueError: If the configuration file is not present.
        ValueError: If n_examples is not a non-zero, positive integer.
    """
    cfg = cfg.data

    print(os.getcwd())
    print(cfg)
    if not cfg:
        raise ValueError("Configuration file must be present.")
    if cfg.n_examples <= 0:
        raise ValueError("n_examples must be a non-zero, positive integer.")
    df = pd.read_csv(cfg.dataset_path, sep=';')
    df_us = df[df['Country']=='United States']
    df = df_us
    df['Reviews per Month'] = df['Reviews per Month'].fillna(0)
    df['ReviewClass'] = df['Reviews per Month'].map(lambda x: numtoClass(x))
    cleancolumn(df,'Summary')
    cleancolumn(df,'Space')
    df['allCleanText'] = df['SummaryClean'] + ' ' + df['SpaceClean']
    df['SummaryClean'] = df['Summary'].fillna('.').map(lambda x : filter_non_english(x))
    df['SummaryClean'] = df['SummaryClean'].map(lambda x : replaceempty(x))
    dffinal = df[['allCleanText','ReviewClass']]
    dffinal.rename(columns={'allCleanText': 'text', 'ReviewClass': 'label'}, inplace=True)
    dffinal = dffinal.sample(n=cfg.n_examples, random_state=cfg.random_state) 
    X = dffinal['text']
    Y = dffinal['label']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state = cfg.random_state)
    dfTraincombined = pd.concat([X_train,y_train], axis = 1)
    dfTestcombined = pd.concat([X_test,y_test], axis = 1)
    logging.debug('os.getcwd(): %s', os.getcwd())
    # dataset_path = os.path.join(hydra.utils.get_original_cwd(), cfg.dataset_path) #Hydra changes cwd
    logging.info('Generating CSV file from dataset...')
    dfTraincombined.to_csv('data/processed/traindata.csv', index=False)
    dfTestcombined.to_csv('data/processed/testdata.csv', index=False)
    #logging.info('Dataset converted to CSV and saved to %s', cfg.dataset_path)
    #generate_csv(cfg.dataset_path, dataset_train)
    logging.info('CSV file generated successfully.')

if __name__=='__main__':
    make_dataset()
