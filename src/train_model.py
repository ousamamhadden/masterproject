import logging
import os
import random

import hydra
import numpy as np
import torch
from happytransformer import HappyTextToText, TTTrainArgs
from omegaconf import DictConfig

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


@hydra.main(config_path="../config", config_name="default_config.yaml", version_base=None)
def train(config: DictConfig) -> None:
    """Train the model using the provided configuration."""
    cfg = config.training
    model = HappyTextToText("t5-small")

    if cfg.metric_tracker != "wandb":  # necessary for unit testing
        args = TTTrainArgs(batch_size=cfg.batch_size, learning_rate=cfg.lr, num_train_epochs=cfg.epochs)
    else:
        args = TTTrainArgs(
            batch_size=cfg.batch_size, report_to=cfg.metric_tracker, learning_rate=cfg.lr, num_train_epochs=cfg.epochs
        )
    set_seed(cfg.seed)
    logging.info("Training model...")
    model.train(cfg.dataset_path, args=args)
    logging.info("Training complete.")
    model.save(cfg.model_path)
    logging.info("Model saved to %s", cfg.model_path)


if __name__ == "__main__":
    train()
