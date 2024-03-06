import csv
import os
from shutil import rmtree

from hydra import compose, initialize

from src.train_model import train


def test_train_model(tmpdir):
    # Create a temporary directory for the model
    temp_dir = tmpdir.mkdir("temp")
    model_dir = os.path.join(temp_dir, "model")

    # Create a mock CSV file
    dataset_path = os.path.join(temp_dir, "dataset.csv")
    with open(dataset_path, "w", newline="") as csvfile:
        fieldnames = ["input", "target"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(2):
            writer.writerow({"input": f"grammar: This are test {i}", "target": f"This is a test {i}"})

    # Run the train function
    with initialize(version_base=None, config_path="../config"):
        # Define a sample configuration
        config = compose(
            config_name="default_config.yaml",
            overrides=[
                f"training.dataset_path={dataset_path}",
                f"training.model_path={model_dir}",
                "training.metric_tracker=False",
                "training.epochs=1",
                "training.batch_size=1",
                "training.lr=0.0001",
                "training.seed=42",
            ],
        )
        train(config)

    # Check if the model directory is created
    assert os.path.exists(model_dir), f"{model_dir} does not exist"

    # Check if the model is saved
    model_path = os.path.join(model_dir, "model.safetensors")
    assert os.path.exists(model_path), f"{model_path} does not exist"

    # Clean up the temporary directories
    rmtree(temp_dir)
