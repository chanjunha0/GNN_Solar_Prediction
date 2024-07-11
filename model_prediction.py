import os

import numpy as np
import pandas as pd
import torch
from torch.nn import Module
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List

from utils import load_model


def make_predictions(
    model: Module, data_loader: DataLoader, device: torch.device
) -> List[float]:
    """
    Generates predictions for the given data using the specified model.

    Args:
        model (Module): The trained model used for generating predictions.
        data_loader (DataLoader): A DataLoader object that provides the data in batches.
        device (torch.device): The device (CPU or GPU) on which the model and data are located.

    Returns:
        List[float]: A list of predicted values.

    Raises:
        None
    """
    predictions = []
    for data in data_loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data.x, data.edge_index).squeeze()
            output = output.cpu().numpy().flatten()
            predictions.extend(output.tolist())
    return predictions


def execute_predictions(
    model_path: str,
    num_node_features: int,
    num_classes: int,
    labels_info_path: str,
    object_start: int,
    object_end: int,
    device: torch.device,
    data_dir: str,
    output_dir: str,
) -> None:
    """
    Executes predictions for a range of objects and saves the results to CSV files.

    Args:
        model_path (str): Path to the model file.
        num_node_features (int): Number of node features in the model.
        num_classes (int): Number of output classes in the model.
        labels_info_path (str): Path to the CSV file containing labels information.
        object_start (int): The starting index of objects for which predictions are made.
        object_end (int): The ending index of objects for which predictions are made.
        device (torch.device): The device (CPU or GPU) on which the model is run.
        data_dir (str): Directory containing the input data files.
        output_dir (str): Directory where the output prediction files will be saved.

    Returns:
        csv: Predictions saved to csv.

    Raises:
        Exception: If any error occurs during the execution.
    """
    labels_info_df = pd.read_csv(labels_info_path, header=None, names=["run_name"])
    model = load_model(model_path, num_node_features, num_classes, device)

    for object_num in range(object_start, object_end + 1):
        object_name = f"run_{object_num}"
        n_results = int(labels_info_df.iloc[object_num - 1]["run_name"])
        new_data = torch.load(os.path.join(data_dir, f"{object_name}.pt"))

        if isinstance(new_data, Data):
            new_data = [new_data]

        data_loader = DataLoader(new_data, batch_size=1, shuffle=False)
        predictions = make_predictions(model, data_loader, device)
        predictions = predictions[:n_results]

        prediction_file_path = os.path.join(
            output_dir, f"{os.path.basename(model_path)}_predictions_{object_name}.csv"
        )
        np.savetxt(prediction_file_path, np.array(predictions), delimiter=",")
