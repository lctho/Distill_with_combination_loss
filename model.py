import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TextClassificationModel:
    """Manages the Transformer model for text classification with combined CE and KL loss."""

    def __init__(self, model_name, num_labels, device='cuda', lr=5e-5):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.kl_div_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.log_softmax = nn.LogSoftmax(dim=1)

    def train(self, dataloader, num_epochs=40, temperature=2.0, alpha=0.05):
        """Trains the model with combined CE and KL loss.

        Args:
            dataloader (DataLoader): Training DataLoader.
            num_epochs (int): Number of training epochs.
            temperature (float): Temperature for KL loss.
            alpha (float): Weight for CE loss (1-alpha for KL loss).
        """
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                self.optimizer.zero_grad()
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in ["soft_labels", "hard_labels"]}
                outputs = self.model(**inputs)
                student_logits = outputs.logits

                hard_labels = batch["hard_labels"].to(self.device)
                ce_loss = self.ce_loss_fn(student_logits, hard_labels)

                soft_labels = batch["soft_labels"].to(self.device)
                student_log_probs = self.log_softmax(student_logits / temperature)
                kl_loss = self.kl_div_loss_fn(student_log_probs, soft_labels)

                loss = alpha * ce_loss + (1 - alpha) * kl_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            self.scheduler.step(avg_loss)

    def save(self, output_dir):
        """Saves the model and tokenizer.

        Args:
            output_dir (str): Directory to save the model.
        """
        self.model.save_pretrained(output_dir)

    def predict(self, dataloader):
        """Runs inference and returns predictions and labels.

        Args:
            dataloader (DataLoader): DataLoader for inference.

        Returns:
            tuple: Lists of predictions and true labels.
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != "hard_labels"}
                labels = batch["hard_labels"].to(self.device)
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        return all_preds, all_labels