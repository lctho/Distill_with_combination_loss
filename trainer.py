import os
from torch.utils.data import DataLoader
from data_processor import DataProcessor
from dataset import TextClassificationDataset
from model import TextClassificationModel
from evaluator import Evaluator


class Trainer:
    """Orchestrates training and testing of the text classification model."""

    def __init__(self, model_name, label_map, device='cuda', batch_size=16):
        self.model_name = model_name
        self.label_map = label_map
        self.device = device
        self.batch_size = batch_size
        self.data_processor = DataProcessor(tokenizer_name=model_name)
        self.evaluator = Evaluator(label_map)

    def train_and_test(self, train_file, test_file, num_epochs=40):
        """Trains and tests the model.

        Args:
            train_file (str): Path to training CSV.
            test_file (str): Path to testing CSV.
            num_epochs (int): Number of training epochs.
        """
        # Load and preprocess training data
        print(f"\nTraining with {train_file} and {self.model_name}")
        train_df = self.data_processor.load_and_preprocess(train_file, is_train=True)
        train_tokenized = self.data_processor.tokenize(train_df["content"].tolist())
        train_soft_labels = train_df["soft_labels"].tolist()
        train_hard_labels = train_df["label_id"].tolist()

        # Create training DataLoader
        train_dataset = TextClassificationDataset(train_tokenized, train_soft_labels, train_hard_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize and train model
        model = TextClassificationModel(self.model_name, num_labels=len(self.label_map), device=self.device, lr=5e-5)
        model.train(train_dataloader, num_epochs=num_epochs)

        # Save model
        output_dir = f"{self.model_name.split('/')[-1]}_{train_file.split('.')[0]}"
        model.save(output_dir)
        self.data_processor.tokenizer.save_pretrained(output_dir)

        # Load and preprocess test data
        print(f"\nTesting model on {test_file}")
        test_df = self.data_processor.load_and_preprocess(test_file, is_train=False)
        test_df["label_id"] = test_df["label"].map(self.label_map)
        if test_df["label_id"].isnull().any():
            missing_labels = test_df[test_df["label_id"].isnull()]["label"].unique()
            raise ValueError(f"Some labels in test data are not in the label_map: {missing_labels}")

        test_tokenized = self.data_processor.tokenize(test_df["content"].tolist())
        test_dataset = TextClassificationDataset(test_tokenized, hard_labels=test_df["label_id"].tolist())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Run inference
        predictions, labels = model.predict(test_dataloader)

        # Evaluate and save results
        output_excel = f"classification_report_{self.model_name.split('/')[-1]}_{test_file.split('.')[0]}.xlsx"
        output_excel = os.path.join('combine_', output_excel)
        report_dict = self.evaluator.evaluate(predictions, labels)
        self.evaluator.save_report(report_dict, output_excel)