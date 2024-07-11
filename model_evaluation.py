import logging
import os

import pandas as pd
from sklearn.metrics import mean_absolute_error

from utils import load_config


def evaluate_model_performance(
    config_path: str, data_dir: str, output_dir: str
) -> pd.DataFrame:
    """
    Evaluates the model performance by calculating MAE, RMSE, and R2 metrics for each run
    and returns a DataFrame with the results.

    Args:
        config_path (str): The file path to the JSON configuration file.
        data_dir (str): The directory containing the actual and predicted labels.
        output_dir (str): The directory where the performance metrics CSV file will be saved.

    Returns:
        pd.DataFrame: A DataFrame containing the performance metrics for each run and the average metrics.

    Raises:
        FileNotFoundError: If any of the CSV files are not found.
        json.JSONDecodeError: If there is an error parsing the JSON file.
    """
    config = load_config(config_path)

    evaluation_config = config["evaluation"]
    model_name = evaluation_config["model_name"]
    start_run = evaluation_config["start_run"]
    end_run = evaluation_config["end_run"]

    runs = [f"run_{i}" for i in range(start_run, end_run + 1)]

    results_df = pd.DataFrame(columns=["Run", "MAE"])

    for run_num in runs:
        actual_label_path = os.path.join(data_dir, f"actual_label_{run_num}.csv")
        predicted_label_path = os.path.join(
            data_dir, f"{model_name}_predictions_{run_num}.csv"
        )

        try:
            label_actual_df = pd.read_csv(actual_label_path, header=None)
            label_predict_df = pd.read_csv(predicted_label_path, header=None)
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise

        actual_values = label_actual_df.iloc[:, 0]
        predicted_values = label_predict_df.iloc[:, 0]

        mae = mean_absolute_error(actual_values, predicted_values)

        new_row = pd.DataFrame({"Run": [run_num], "MAE": [mae]})
        results_df = pd.concat([results_df, new_row], ignore_index=True)

    avg_metrics = results_df.mean(numeric_only=True)
    avg_row = pd.DataFrame({"Run": ["Average"], "MAE": [avg_metrics["MAE"]]})

    results_df = pd.concat([results_df, avg_row], ignore_index=True)

    print(results_df)

    performance_file_path = os.path.join(
        output_dir, f"model_performance_metrics_{model_name}.csv"
    )
    results_df.to_csv(performance_file_path, index=False)

    return results_df
