import json
import logging

import torch
from model_classes import SimplifiedGCN
from torch.nn import Module
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a configuration file in JSON format and returns it as a dictionary.

    Args:
        config_path (str): The file path to the JSON configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration data.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        json.JSONDecodeError: If there is an error parsing the JSON file.
    """
    try:
        with open(config_path, "r") as file:
            config = json.load(file)
        return config
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON configuration file: {e}")
        raise


def load_model(
    model_path: str, num_node_features: int, num_classes: int, device: torch.device
) -> Module:
    """
    Loads a trained model from the specified path and prepares it for evaluation.

    Args:
        model_path (str): The file path to the model's state dictionary.
        num_node_features (int): The number of features per node in the model.
        num_classes (int): The number of output classes for the model.
        device (torch.device): The device (CPU or GPU) on which the model will be run.

    Returns:
        Module: The loaded and evaluated model.

    Raises:
        Exception: If there is an error loading the model.
    """
    try:
        model = SimplifiedGCN(num_node_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise
