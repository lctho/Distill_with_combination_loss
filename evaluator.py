import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tabulate import tabulate


class Evaluator:
    """Computes and saves evaluation metrics for text classification."""

    def __init__(self, label_map):
        self.label_map = label_map
        self.label_names = list(label_map.keys())

    def evaluate(self, predictions, labels):
        """Computes per-class and overall metrics.

        Args:
            predictions (list): Model predictions.
            labels (list): True labels.

        Returns:
            dict: Classification report with micro avg.
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, labels=list(self.label_map.values())
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted"
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            labels, predictions, average="micro"
        )
        accuracy = accuracy_score(labels, predictions)

        # Print per-class metrics
        print("\nPer-class metrics:")
        for label, idx in self.label_map.items():
            print(f"Label '{label}' (ID {idx}):")
            print(f"  Precision: {precision[idx]:.4f}")
            print(f"  Recall: {recall[idx]:.4f}")
            print(f"  F1 Score: {f1[idx]:.4f}")
            print(f"  Support: {support[idx]}")

        # Print overall metrics
        print("\nOverall metrics:")
        print(
            f"Weighted Avg - Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1 Score: {f1_weighted:.4f}")
        print(f"Micro Avg - Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1 Score: {f1_micro:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Total samples: {len(labels)}")

        # Print table
        headers = ["Label", "Precision", "Recall", "F1-Score"]
        table_data = []
        for label, idx in self.label_map.items():
            table_data.append([label, precision[idx], recall[idx], f1[idx]])
        table_data.append(["Weighted Avg", precision_weighted, recall_weighted, f1_weighted])
        table_data.append(["Micro Avg", precision_micro, recall_micro, f1_micro])
        print("\nEvaluation Results:")
        print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))

        # Classification report
        report_dict = classification_report(labels, predictions, target_names=self.label_names, output_dict=True)
        report_dict["micro avg"] = {
            "precision": precision_micro,
            "recall": recall_micro,
            "f1-score": f1_micro,
            "support": len(labels)
        }
        report_dict["accuracy"] = {
            "precision": accuracy,
            "recall": accuracy,
            "f1-score": accuracy,
            "support": len(labels)
        }

        return report_dict

    def save_report(self, report_dict, output_excel):
        """Saves classification report to Excel.

        Args:
            report_dict (dict): Classification report dictionary.
            output_excel (str): Path to save Excel file.
        """
        excel_data = []
        for label, metrics in report_dict.items():
            excel_data.append({
                "Label": label,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1-score"],
                "Support": metrics["support"]
            })
        report_df = pd.DataFrame(excel_data)
        report_df.to_excel(output_excel, index=False)
        print(f"\nClassification report saved to {output_excel}")