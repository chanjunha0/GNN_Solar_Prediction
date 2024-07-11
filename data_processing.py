import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from utils import load_config


def process_run(
    run_num: str, num_labeled_nodes: int, config: Dict[str, Any], pipeline_mode: str
) -> None:
    """
    Process a single run, generating Data objects for training or prediction.

    Args:
        run_num (str): The run number identifier.
        num_labeled_nodes (int): The number of labeled nodes.
        config (Dict[str, Any]): The configuration dictionary.
        pipeline_mode (str): The mode of the pipeline, either 'training' or 'prediction'.

    Returns:
        None
    """
    file_names = config["data_processing"]["file_names"]
    append_list = config["data_processing"]["append_list"]
    df_material_map = config["data_processing"]["df_material_map"]
    df_vertex_names = config["data_processing"]["df_vertex_names"]
    material_list = config["data_processing"]["material_list"]
    node_features_columns = config["data_processing"]["node_features"]

    base_path = f"data/csv_{pipeline_mode}/{run_num}"
    filename = f"data/torch_{pipeline_mode}/{run_num}.pt"

    def format_and_insert_id_column(
        df: pd.DataFrame, id_base_name: str
    ) -> pd.DataFrame:
        df.reset_index(drop=True, inplace=True)
        df[id_base_name] = [f"{id_base_name}_{i + 1}" for i in df.index]
        df.insert(0, id_base_name, df.pop(id_base_name))
        return df

    def load_csv_files_as_dict(
        file_path: str, file_names: List[str]
    ) -> Dict[str, pd.DataFrame]:
        dataframes = {}
        for file_name in file_names:
            file_path_full = os.path.join(file_path, f"{file_name}.csv")
            if os.path.getsize(file_path_full) > 0:
                df_name = f"{file_name}_df"
                dataframes[df_name] = pd.read_csv(file_path_full, header=None)
        return dataframes

    def append_material_properties(
        dfs: Dict[str, pd.DataFrame],
        material_df: pd.DataFrame,
        df_material_map: Dict[str, str],
    ) -> Dict[str, pd.DataFrame]:
        for df_name, material_name in df_material_map.items():
            new_df_name = f"{df_name}_append"
            if df_name in dfs and not dfs[df_name].empty:
                material_row = material_df[
                    material_df["material_name"] == material_name
                ].drop("material_name", axis=1)
                repeated_material = pd.concat(
                    [material_row] * len(dfs[df_name]), ignore_index=True
                )
                dfs[new_df_name] = pd.concat(
                    [
                        dfs[df_name].reset_index(drop=True),
                        repeated_material.reset_index(drop=True),
                    ],
                    axis=1,
                )
        return dfs

    def combine_dataframes_in_order(
        dfs: Dict[str, pd.DataFrame], append_list: List[str]
    ) -> pd.DataFrame:
        combined_df = pd.concat(
            [dfs[df_name] for df_name in append_list if df_name in dfs],
            ignore_index=True,
        )
        return combined_df

    def extract_values_to_dict(
        dfs: Dict[str, pd.DataFrame], df_names: List[str]
    ) -> Dict[str, Any]:
        return {
            df_name: dfs[df_name].iloc[0, 0]
            for df_name in df_names
            if df_name in dfs and not dfs[df_name].empty
        }

    def map_sensor_to_vertex(
        sensor_length: int, values_dict: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        mapped_dfs = {}
        for material, length in values_dict.items():
            material_name = material.split("_df")[0]
            data = [
                (f"sensor_id_{sensor_id}", f"{material_name}_{i}")
                for sensor_id in range(1, sensor_length + 1)
                for i in range(1, length + 1)
            ]
            mapped_df = pd.DataFrame(data, columns=["sensor_id", "vertex_id"])
            mapped_dfs[material_name] = mapped_df
        return mapped_dfs

    def append_distance_to_mapped_dfs(
        mapped_dfs: Dict[str, pd.DataFrame],
        dfs: Dict[str, pd.DataFrame],
        material_list: List[str],
    ) -> Dict[str, pd.DataFrame]:
        for material in material_list:
            distance_df_name = f"distance_{material}_df"
            mapped_df_name = f"vertex_length_{material}"
            if distance_df_name in dfs and mapped_df_name in mapped_dfs:
                distance_df = dfs[distance_df_name]
                mapped_df = mapped_dfs[mapped_df_name]
                if len(distance_df) >= len(mapped_df):
                    mapped_df["distance"] = distance_df.iloc[: len(mapped_df), 0].values
                else:
                    logging.warning(
                        f"Warning: Not enough distance values for {material}, distances not appended."
                    )
                mapped_dfs[mapped_df_name] = mapped_df
        return mapped_dfs

    # Load CSV files and material properties
    dfs = load_csv_files_as_dict(base_path, file_names)
    material_df = pd.read_csv("data/material_library.csv")
    dfs["material"] = material_df
    dfs = append_material_properties(dfs, material_df, df_material_map)

    # Combine DataFrames and extract values
    building_df = combine_dataframes_in_order(dfs, append_list)
    values_dict = extract_values_to_dict(dfs, df_vertex_names)
    sensor_length = int(dfs["sensor_length_df"].iloc[0, 0])

    # Map sensors to vertices and append distances
    mapped_dfs = map_sensor_to_vertex(sensor_length, values_dict)
    final_dfs = append_distance_to_mapped_dfs(mapped_dfs, dfs, material_list)

    # Concatenate DataFrames in alphabetical order
    sorted_keys = sorted(final_dfs.keys())
    edge_df = pd.concat([final_dfs[key] for key in sorted_keys], ignore_index=True)

    sensor_df = format_and_insert_id_column(dfs["sensor_df"], "sensor_id")
    building_df = format_and_insert_id_column(building_df, "vertex_id")
    label_df = format_and_insert_id_column(dfs["label_df"], "sensor_id")

    sensor_df.rename(
        columns={0: "sensor_x_", 1: "sensor_y_", 2: "sensor_z_"}, inplace=True
    )
    building_df.rename(
        columns={0: "building_x", 1: "building_y", 2: "building_z"}, inplace=True
    )
    label_df.columns = ["sensor_id", "hb_solar_radiation"]

    angle_df = dfs["angle_degree_df"]
    normal_df = dfs["normal_df"].apply(
        lambda x: x.map(lambda x: x.strip("{}") if isinstance(x, str) else x)
    )

    sensor_df["type"] = "sensor"
    building_df["type"] = "vertex"
    sensor_df["angle"] = angle_df.iloc[:, 0]
    sensor_df["normal_x"] = normal_df.iloc[:, 0]
    sensor_df["normal_y"] = normal_df.iloc[:, 1]
    sensor_df["normal_z"] = normal_df.iloc[:, 2]

    all_nodes_df = pd.concat(
        [
            sensor_df.assign(index=range(0, len(sensor_df))),
            building_df.assign(
                index=range(len(sensor_df), len(sensor_df) + len(building_df))
            ),
        ]
    )
    all_nodes_df["type_flag"] = all_nodes_df["type"].apply(
        lambda x: 1 if x == "sensor" else 0
    )
    node_features = all_nodes_df[node_features_columns].fillna(0).values
    node_features_numeric = np.array(
        [[float(val) for val in row] for row in node_features]
    )
    x = torch.tensor(node_features_numeric, dtype=torch.float)

    sensor_ids = sensor_df["sensor_id"].unique()
    vertex_ids = building_df["vertex_id"].unique()

    sensor_index = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}
    vertex_index = {
        vertex_id: i + len(sensor_index) for i, vertex_id in enumerate(vertex_ids)
    }

    edge_df["adjusted_vertex_id"] = edge_df["vertex_id"].apply(
        lambda x: "vertex_id_" + x.split("_")[-1]
    )
    edge_index_list = edge_df.apply(
        lambda row: [
            sensor_index.get(row["sensor_id"], -1),
            vertex_index.get(row["adjusted_vertex_id"], -1),
        ],
        axis=1,
    )
    filtered_edge_index_list = [pair for pair in edge_index_list if -1 not in pair]
    edge_index = (
        torch.tensor(filtered_edge_index_list, dtype=torch.long).t().contiguous()
    )
    edge_attr = torch.tensor(edge_df[["distance"]].values, dtype=torch.float)

    label_df["hb_solar_radiation"] = label_df["hb_solar_radiation"].astype(float)
    label_df["index"] = label_df["sensor_id"].map(sensor_index)
    labels = torch.zeros(len(label_df), dtype=torch.float)
    labels[label_df["index"]] = torch.tensor(
        label_df["hb_solar_radiation"].values, dtype=torch.float
    )

    train_mask = torch.zeros(len(sensor_ids), dtype=torch.bool)
    test_mask = torch.zeros(len(sensor_ids), dtype=torch.bool)

    labeled_indices = torch.arange(num_labeled_nodes)
    np.random.shuffle(labeled_indices.numpy())
    split_point = int(len(labeled_indices) * 0.8)

    train_indices = labeled_indices[:split_point]
    test_indices = labeled_indices[split_point:]

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    if pipeline_mode == "training":
        data_training = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels,
            train_mask=train_mask,
            test_mask=test_mask,
        )
        torch.save(data_training, filename)
    elif pipeline_mode == "prediction":
        data_predict = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        torch.save(data_predict, filename)


def run_data_processing(config_path: str, pipeline_mode: str) -> None:
    """
    Runs the data processing pipeline for a range of runs.

    Args:
        config_path (str): Path to the configuration file.
        pipeline_mode (str): The mode of the pipeline, either 'training' or 'prediction'.

    Returns:
        None
    """
    config = load_config(config_path)

    labels_info_path = f"data/labels_info_{pipeline_mode}.csv"
    labels_info = pd.read_csv(labels_info_path, header=None, usecols=[0])
    num_labeled_nodes_list = [int(x) for x in labels_info[0].tolist()]

    for i in range(1, 55):
        run_number = f"run_{i}"
        num_labeled_nodes = num_labeled_nodes_list[i - 1]
        logging.info(f"Processing {run_number} with {num_labeled_nodes} labeled nodes.")
        process_run(run_number, num_labeled_nodes, config, pipeline_mode)
        logging.info(f"Completed processing {run_number}.")
