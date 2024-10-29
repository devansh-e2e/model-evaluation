import pandas as pd
from datasets import load_dataset


class CustomDatasetLoader:
    def __init__(self, dataset_name, input_field, ground_truth_field, split="train"):
        """
        Initialize the dataset loader with the Hugging Face dataset name, input and ground truth field names.

        Parameters:
            dataset_name (str): Name of the dataset on Hugging Face.
            input_field (str): Name of the input field in the dataset.
            ground_truth_field (str): Name of the ground truth field in the dataset.
            split (str): The split of the dataset to load (default is "train").
        """
        self.dataset_name = dataset_name
        self.input_field = input_field
        self.ground_truth_field = ground_truth_field
        self.split = split
        self.dataset = None

    def load_data(self):
        """Load the dataset and store it."""
        self.dataset = load_dataset(self.dataset_name, split=self.split)
        self.validate_fields()

    def validate_fields(self):
        """Ensure that the specified input and ground truth fields exist in the dataset."""
        if self.dataset is None:
            raise ValueError("Dataset is not loaded. Please load the dataset first.")
        if self.input_field not in self.dataset.column_names:
            raise ValueError(f"Input field '{self.input_field}' not found in the dataset columns.")
        if self.ground_truth_field not in self.dataset.column_names:
            raise ValueError(f"Ground truth field '{self.ground_truth_field}' not found in the dataset columns.")

    def to_dataframe(self):
        """
        Convert the dataset to a pandas DataFrame with specified columns.

        Returns:
            pd.DataFrame: A DataFrame containing the inputs and ground truths.
        """
        if self.dataset is None:
            self.load_data()
        data = {
            "inputs": self.dataset[self.input_field],
            "ground_truth": self.dataset[self.ground_truth_field]
        }
        eval_df = pd.DataFrame(data)
        return eval_df

# Example usage:
# loader = CustomDatasetLoader("dataset_name", "input_field", "output_field")
# eval_df = loader.to_dataframe()
# print(eval_df.head())
