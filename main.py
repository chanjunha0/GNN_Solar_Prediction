import logging
import os
import torch

from data_processing import run_data_processing
from model_classes import SimplifiedGCN
from model_evaluation import evaluate_model_performance
from model_prediction import execute_predictions
from model_training import run_training
from utils import load_config


# Load configuration
config_path = "config.json"
config = load_config(config_path)
run_parameters = config["run_parameters"]


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load constants from config
model_path = os.path.join("data", config["evaluation"]["model_name"] + ".pth")
num_node_features = 16
num_classes = 1

labels_info_path = "data/labels_info_prediction.csv"
data_dir = "data"
output_dir = "data"

pipeline_mode = "training"

object_start = config["evaluation"]["start_run"]
object_end = config["evaluation"]["end_run"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data Processing Pipeline
if run_parameters["data_processing"]:
    logging.info("Starting data processing pipeline...")
    run_data_processing(config_path, pipeline_mode)
    logging.info("Data processing pipeline completed successfully.")

# Model Training Pipeline
if run_parameters["model_training"]:
    logging.info("Starting model training pipeline...")
    run_training(
        folder_path=config["training"]["folder_path"],
        labels_info_path=config["training"]["labels_info_path"],
        model_path=config["training"]["model_path"],
        model_type=SimplifiedGCN,
        total_objects=config["training"]["total_objects"],
        num_epochs=config["training"]["num_epochs"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    logging.info("Model training pipeline completed successfully.")

# Model Prediction Pipeline
if run_parameters["model_prediction"]:
    logging.info("Starting prediction pipeline...")
    try:
        execute_predictions(
            model_path,
            num_node_features,
            num_classes,
            labels_info_path,
            object_start,
            object_end,
            device,
            data_dir,
            output_dir,
        )
        logging.info("Prediction pipeline completed successfully.")
    except FileNotFoundError as e:
        logging.error(f"Error in prediction pipeline: {e}")


# Model Evaluation Pipeline
if run_parameters["model_evaluation"]:
    logging.info("Evaluating model performance...")
    try:
        results_df = evaluate_model_performance("config.json", data_dir, output_dir)
        logging.info("Model performance evaluation completed.")
    except FileNotFoundError as e:
        logging.error(f"Error in model performance evaluation: {e}")

logging.info("Solar Radiation GNN Pipeline Completed.")
