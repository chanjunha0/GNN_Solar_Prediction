import logging
import os
import time
from typing import Any, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data


def load_graph_data_objects(
    folder_path: str, ranges: List[Tuple[int, int]]
) -> List[List[Data]]:
    """
    Load graph data objects from a folder based on specified ranges.

    Args:
        folder_path (str): Path to the folder containing graph data objects.
        ranges (List[Tuple[int, int]]): List of tuples where each tuple contains the start and end run numbers.

    Returns:
        List[List[Data]]: A list of lists, where each inner list contains graph data objects for a specific town.
    """
    graph_data_objects_by_town = []

    for start_run, end_run in ranges:
        town_simulations = [
            torch.load(os.path.join(folder_path, f"run_{run_number}.pt"))
            for run_number in range(start_run, end_run + 1)
            if os.path.exists(os.path.join(folder_path, f"run_{run_number}.pt"))
        ]
        graph_data_objects_by_town.append(town_simulations)

    return graph_data_objects_by_town


def load_or_initialize_model(
    model_path: str, model_class: type, *model_args: Any, **model_kwargs: Any
) -> Any:
    """
    Load a previously trained model or initialize a new one.

    Args:
        model_path (str): Path to the model file.
        model_class (type): The model class to be instantiated.
        *model_args: Variable length argument list for the model class.
        **model_kwargs: Arbitrary keyword arguments for the model class.

    Returns:
        Any: An instance of the model class, either loaded with pre-trained weights or newly initialized.
    """
    model = model_class(*model_args, **model_kwargs)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        logging.info("Loaded previously trained model.")
    else:
        logging.info("Initialized new model.")
    return model


def load_num_labeled_nodes(labels_info_path: str) -> List[int]:
    """
    Load the number of labeled nodes from a CSV file.

    Args:
        labels_info_path (str): Path to the CSV file containing label information.

    Returns:
        List[int]: A list of labeled nodes.
    """
    labels_info = pd.read_csv(labels_info_path, usecols=[0])
    return labels_info.iloc[:, 0].tolist()


def train(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: Data
) -> float:
    """
    Train the model on a given batch of data.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        batch (Data): The batch of data for training.

    Returns:
        float: The training loss value.
    """
    model.train()
    optimizer.zero_grad()

    out = model(batch.x, batch.edge_index).squeeze()
    valid_out = out[: batch.train_mask.sum().item()]
    valid_labels = batch.y[batch.train_mask]

    loss = F.mse_loss(valid_out, valid_labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: torch.nn.Module, batch: Data) -> float:
    """
    Evaluate the model on a given batch of data.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        batch (Data): The batch of data for testing.

    Returns:
        float: The test loss value.
    """
    model.eval()
    with torch.no_grad():
        out = model(batch.x, batch.edge_index).squeeze()
        test_out = out[batch.test_mask]
        valid_labels = batch.y[batch.test_mask]

        test_loss = F.mse_loss(test_out, valid_labels).item()

    return test_loss


def run_training(
    folder_path: str,
    labels_info_path: str,
    model_path: str,
    model_type: type,
    total_objects: int,
    num_epochs: int = 50,
    learning_rate: float = 0.01,
    weight_decay: float = 0.01,
) -> None:
    """
    Runs the training process.

    Args:
        folder_path (str): Path to the folder containing graph data objects.
        labels_info_path (str): Path to the CSV file containing label information.
        model_path (str): Path to the model file.
        model_type (type): The model class to be instantiated.
        total_objects (int): The total number of graph data objects.
        num_epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.

    Returns:
        None
    """
    logging.info("Starting the training process.")

    ranges = [(i, i) for i in range(1, total_objects + 1)]

    graph_data_objects = load_graph_data_objects(folder_path, ranges)

    if graph_data_objects and graph_data_objects[0]:
        num_node_features = graph_data_objects[0][0].num_node_features
        logging.info(f"Number of node features: {num_node_features}")
    else:
        logging.error("Error: No graph data objects loaded.")
        return

    num_classes = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_or_initialize_model(
        model_path, model_type, num_node_features, num_classes
    ).to(device)
    logging.info(f"Model {model_type.__name__} is ready for training.")
    if device.type == "cuda":
        logging.info(f"Model is using GPU: {torch.cuda.get_device_name(device)}")
    else:
        logging.info("Model is using CPU")

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    test_losses, average_epoch_losses = [], []

    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        epoch_loss = 0
        for data_list in graph_data_objects:
            for data in data_list:
                data = data.to(device)
                train_loss = train(model, optimizer, data)
                epoch_loss += train_loss
        average_loss = epoch_loss / sum(
            len(data_list) for data_list in graph_data_objects
        )
        average_epoch_losses.append(average_loss)

        test_epoch_loss = 0
        with torch.no_grad():
            for data_list in graph_data_objects:
                for data in data_list:
                    data = data.to(device)
                    test_loss = test(model, data)
                    test_epoch_loss += test_loss
        average_test_loss = test_epoch_loss / sum(
            len(data_list) for data_list in graph_data_objects
        )
        test_losses.append(average_test_loss)

        epoch_runtime = time.time() - epoch_start_time
        total_elapsed_time = time.time() - total_start_time

        logging.info(
            f"Epoch {epoch+1}, Average Train Loss: {average_loss:.2f}, Average Test Loss: {average_test_loss:.2f}, Runtime: {epoch_runtime:.2f} seconds, Total Elapsed Time: {(total_elapsed_time)/60:.2f} minutes"
        )

    torch.save(model.state_dict(), model_path)

    end_time = time.time()
    training_duration = end_time - total_start_time
    minutes, seconds = divmod(training_duration, 60)
    hours, minutes = divmod(minutes, 60)
    logging.info(
        f"Training took {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds."
    )
