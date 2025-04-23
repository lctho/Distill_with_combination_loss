import json


def load_label_map(json_path="data_label.json"):
    """Loads label mapping from a JSON file.

    Args:
        json_path (str): Path to JSON file.

    Returns:
        dict: Label mapping (e.g., {"ordering": 0, "others": 13}).
    """
    with open(json_path, "r") as file:
        label_map = json.load(file)
    return label_map