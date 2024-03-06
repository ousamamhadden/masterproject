import csv
import os

from hydra import compose, initialize
from omegaconf import OmegaConf

from src.data.make_dataset import generate_csv, make_dataset


def test_generate_csv(tmpdir) -> None:
    # Create a temporary directory for the CSV file
    temp_dir = tmpdir.mkdir("temp")
    csv_path = os.path.join(temp_dir, "test.csv")

    # Define a sample dataset
    class SampleDataset:
        def __iter__(self):
            yield {"input": "example input", "output": "example output"}

    # Generate the CSV file
    dataset = SampleDataset()
    generate_csv(csv_path, dataset)

    # Read the CSV file and check its contents
    with open(csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)
        assert rows[0] == ["input", "target"]
        assert rows[1] == ["grammar: example input", "example output"]

def test_make_dataset(tmpdir) -> None:
    # Create a temporary directory for the CSV file
    temp_dir = tmpdir.mkdir("temp")
    csv_path = os.path.join(temp_dir, "test.csv")

    with initialize(version_base=None, config_path="../config"):
        # Run the make_dataset function
        config = compose(config_name="default_config", overrides=["data.n_examples=1", f"data.dataset_path={csv_path}"])
        print(f"configuration: \n {OmegaConf.to_yaml(config)}")
        make_dataset(config)

        # Check if the CSV file is generated
        assert os.path.exists(csv_path)

        # Read the CSV file and check its contents
        with open(csv_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            assert rows[0] == ["input", "target"]
            assert len(rows) == config.data.n_examples + 1  # +1 for the header row
