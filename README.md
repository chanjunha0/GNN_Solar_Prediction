# A Rapid Prediction Model for Urban Building Facade Solar Radiation using Graph Neural Networks

This repository contains scripts and configurations for data processing, model training, prediction, and evaluation for a machine learning project involving training a model to predict facade solar radiation using Graph Neural Networks.


- Address the challenges posed by the Urban Heat Island (UHI) warming effect in Singapore.
- The goal is to architect a new solution for climatic simulations that is more accurate and efficient.
- The methodology involves utilsing graph neural networks (GNNs) as a model to interpret BREP structures as graph data, allowing for rapid and accurate solar radiation simulations.

## Repository Structure
```
├── data/
│   ├── csv_training/
│   ├── csv_prediction/
│   ├── torch_data_object_training/
│   └── torch_data_object_prediction/
├── main.py
├── model_training.py
├── data_processing.py
├── model_classes.py
├── model_evaluation.py
├── model_prediction.py
├── config.json
├── README.md
└── utils.py
```


## Configuration File

The `config.json` file contains configuration details used across various scripts in the repository. The `run_parameters` section specifies which parts of the pipeline to execute.

## Scripts Overview

### main.py

The entry point for the entire pipeline. It loads the configuration and runs the specified parts of the pipeline based on the `run_parameters` in `config.json`.

### data_processing.py

Handles the preprocessing of raw CSV data into format suitable for training or prediction. It processes each run, generates data objects, and saves them as `.pt` files for use in model training or prediction.

### model_training.py

Trains the graph-based model using the processed data objects. It supports loading existing models or initializing new ones and tracks training and test losses over epochs.

### model_prediction.py

Uses the trained model to make predictions on new data. The predictions are saved as CSV files for later evaluation.

### model_evaluation.py

Evaluates the performance of the model by comparing the predictions against actual labels. It calculates metrics like Mean Absolute Error (MAE) and saves the results.

### model_classes.py

Defines the model architectures used in the project. For example, it includes a simplified Graph Convolutional Network (GCN) implementation.

### utils.py

Contains utility functions used across different scripts, such as loading configuration files.

## Solar Radiation Simulation Data
The solar radiation data generated from the environmental simulations and  used to execute this pipeline can be downloaded from the following links. Place them in the folder `data`:
1. https://drive.google.com/drive/folders/16JFZfTpWTcYRpeOXEP0ilPYCk_d2kaY6?usp=sharing
2. https://drive.google.com/drive/folders/1T1vCJtHAmh29IWMf-PJp_V5pUw-KsbW8?usp=sharing

## Grasshopper Simulation Methodology

### Overall Script Logic
1. Import 3D model context of urban town.
2. Modify the urban context by height, rotation, and translation.
3. Apply material modifiers: concrete, wood, glass.
4. Run Radiance simulation for a grid size of 5m.
5. Export features from simulations:

| Feature                     | Description                          |
|-----------------------------|--------------------------------------|
| XYZ sensor coordinates      | Coordinates of each sensor           |
| Sensor length               | Length of each sensor                |
| Building vertex coordinates | Coordinates of each building vertex  |
| Vertex length               | Length of each vertex                |
| Sensor to vertex distance   | Distance from sensor to vertex       |
| Embedded material properties| Properties of materials used         |
| Normal vector of sensor     | Normal vector direction of sensor    |
| Angle to north vector       | Angle between sensor and north vector|


## How to Run

1. **Set Up Configuration:**
   - Update `config.json` with the appropriate settings for your data and model.

2. **Run Data Processing:**
   - Ensure `data_processing` is set to `true` in the `run_parameters` section of `config.json`.
   - Execute the main script:
     ```
     python main.py
     ```

3. **Run Model Training:**
   - Set `model_training` to `true` in the `run_parameters` section of `config.json`.
   - Execute the main script:
     ```
     python main.py
     ```

4. **Run Model Prediction:**
   - Set `model_prediction` to `true` in the `run_parameters` section of `config.json`.
   - Execute the main script:
     ```
     python main.py
     ```

5. **Run Model Evaluation:**
   - Set `model_evaluation` to `true` in the `run_parameters` section of `config.json`.
   - Execute the main script:
     ```
     python main.py
     ```

## Logging

Logging is used throughout the scripts to provide detailed information about the execution process. Check the console output or log files for detailed execution traces and any potential issues.

## Dependencies

- matplotlib
- numpy
- pandas
- PyTorch
- Python 3.8+
- scikit-learn
- torch_geometric

Install the dependencies using:
```
pip install -r requirements.txt
```


## Docker Setup

To run this project using Docker, follow the steps below:

### 1. Ensure you have Docker and Docker Compose installed.

- [Docker Installation Instructions](https://docs.docker.com/get-docker/)
- [Docker Compose Installation Instructions](https://docs.docker.com/compose/install/)

### 2. Build and run the Docker container:

Use the following command in your terminal to build and run the Docker container:

```sh
docker-compose up --build
```