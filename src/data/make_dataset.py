import csv
import logging
import os

import hydra
from datasets import IterableDataset, load_dataset
from omegaconf import DictConfig

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def generate_csv(csv_path: str | os.PathLike,
                 dataset: IterableDataset
                 ) -> None:
    """Generate a CSV file from the given dataset that conforms with
    HappyTransformers Text2Text transformers training data format.

    Args:
        csv_path (str | os.PathLike): The path to the CSV file to be generated.
        dataset (IterableDataset): The dataset to be converted to CSV.

    Returns:
        None
    """
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["input", "target"])
        for example in dataset:
            input_text = f"grammar: {example['input']}"
            target_text = example["output"]
            writer.writerow([input_text, target_text])

    logging.info('Dataset converted to CSV and saved to %s', csv_path)

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

    if not cfg:
        raise ValueError("Configuration file must be present.")
    if cfg.n_examples <= 0:
        raise ValueError("n_examples must be a non-zero, positive integer.")
    dataset_train = load_dataset('liweili/c4_200m', split='train', streaming=True, trust_remote_code=True)
    dataset_train = dataset_train.take(cfg.n_examples)
    logging.debug('os.getcwd(): %s', os.getcwd())
    # dataset_path = os.path.join(hydra.utils.get_original_cwd(), cfg.dataset_path) #Hydra changes cwd
    logging.info('Generating CSV file from dataset...')
    generate_csv(cfg.dataset_path, dataset_train)
    logging.info('CSV file generated successfully.')

if __name__=='__main__':
    make_dataset()
