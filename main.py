import os
from trainer import Trainer
from utils import load_label_map
import torch

def main():
    # Configuration
    train_files = ['trainfile1.csv']  # Add more files as needed
    test_files = ['testfile1.csv']  # Add more files as needed
    algorithms = ['uitnlp/visobert']  # Add more models as needed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load label map
    label_map = load_label_map("data_label1.json")

    # Ensure output directory exists
    os.makedirs('combine_', exist_ok=True)

    # Training and testing loop
    for train_file, test_file in zip(train_files, test_files):
        for model_name in algorithms:
            trainer = Trainer(model_name, label_map, device=device, batch_size=16)
            trainer.train_and_test(train_file, test_file, num_epochs=1)


if __name__ == "__main__":
    main()