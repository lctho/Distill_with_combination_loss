import pandas as pd
import ast
from transformers import AutoTokenizer


class DataProcessor:
    """Handles data loading, preprocessing, and tokenization for text classification."""

    def __init__(self, tokenizer_name='uitnlp/visobert', max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def load_and_preprocess(self, file_path, is_train=True):
        """Loads and preprocesses CSV data.

        Args:
            file_path (str): Path to CSV file.
            is_train (bool): Whether the file is for training (includes soft_labels).

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        df = pd.read_csv(file_path)
        df["content"] = df["content"].fillna("").astype(str)
        if df["content"].str.strip().eq("").any():
            print(f"Warning: Empty content entries in {file_path}")

        if is_train:
            if "soft_labels" in df.columns:
                df["soft_labels"] = df["soft_labels"].apply(ast.literal_eval)
            else:
                raise ValueError("Training file must contain 'soft_labels' column.")
        else:
            if not all(col in df.columns for col in ["content", "label"]):
                raise ValueError("Test file must contain 'content' and 'label' columns.")

        return df

    def tokenize(self, texts):
        """Tokenizes text data.

        Args:
            texts (list): List of text strings.

        Returns:
            dict: Tokenized data (input_ids, attention_mask, etc.).
        """
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )