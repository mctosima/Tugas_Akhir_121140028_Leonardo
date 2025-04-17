# Fall Detection using Graph Convolutional Network Dased on Skeleton Landmarks Mediapipe

## Getting Started
1. Clone this repository
2. Create a virtual environment with Python 3.10. Activate the virtual environment
3. Install the required packages
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Project
Activate the virtual environment and run the following command:
```bash
python train_gcn_kfold.py
```

## How to testing the model
1. copy the best model weights from the `pth` folder to the `testing` folder
2. Run the following command:
```bash
python "testing/test_gcn_kfold.py"
```

## Project Structure
- `data-skeleton`: Contains the extracted skeleton landmarks from the dataset
- `extract_frame_program`: Contains the program to extract the frames from the video dataset
- `skeleton-preview`: Contains the preview of the skeleton landmarks in .jpg format
- `splits`: Contains the train and validation splits path (implementing 5-Fold splits) in .json format
- `model`: Contains the PyTorch model
- `pth`: Contains the trained model weights
- `testing`: Contains the testing data (skeleton landmarks) and the testing script

## `.py` Files
- `extract_skeleton.py`: Extract the skeleton landmarks from the dataset.
- `func_distance_features.py`: A function to calculate the distance features from the skeleton landmarks.
- `func_lm_to_graph.py`: A function to convert the skeleton landmarks to a graph representation.
- `utils.py`: My handy tools to select the GPU
- `k_fold_datareader.py`: A custom PyTorch program that contains the functions to read the skeleton landmarks, generate 5-fold splits then save it to folder splits, and verify those 5-Fold split files.
- `k_fold_dataset.py`: A custom PyTorch class that contains the functions to read the 5-Fold split files and generate the graph representation of the skeleton landmarks.
- `model/gcn_model.py`: The Graph Convolutional Network model for the skeleton landmarks fall detection.
- `train_gcn_kfold.py`: The main script for the project: including read, load dataset and train the Graph Convolutional Network model.
- `utils.py`: A custom PyTorch function to detect any appropriate gpu that can be used for faster training.
- `testing/extract_skeleton.py`: A script to extract the skeleton landmarks from the testing dataset.
- `testing/video_frame_extract.py`: A script to extract the frames from the test video dataset.
- `testing/prediction_v2.py`: A script to predict the fall detection using the trained model.
