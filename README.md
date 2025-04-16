## Getting Started
1. Clone this repository
2. Create an empty folder named `data-kaggle`, `data-skeleton` and `skeleton-preview`
3. Copy the dataset from Kaggle to the `data-kaggle` folder
    - Inside `data-kaggle` folder, there should be a folder named `train`, `test`, and `sample_submission.csv`
4. Create a virtual environment with Python 3.10. Activate the virtual environment
5. Install the required packages
    ```bash
    pip install -r requirements.txt
    ```


## Project Structure
- `data-kaggle`: Contains the dataset from Kaggle
- `data-skeleton`: Contains the extracted skeleton landmarks from the dataset
- `skeleton-preview`: Contains the preview of the skeleton landmarks in .jpg format
- `splits`: Contains the train and validation splits path in .json format
    - `splits/LOSO_split.json`: Leave-One-Subject-Out split. Satu subjek dijadikan validation set
    - `splits/Mix_split.json`: Untuk setiap scene pada `fall` dan `non_fall`, ambil dari satu subjek untuk dijadikan validation set
- `model`: Contains the pre-trained PyTorch model

## `.py` Files
- `extract_skeleton.py`: Extract the skeleton landmarks from the dataset
- `func_distance_features.py`: A function to calculate the distance features from the skeleton landmarks
- `func_lm_to_graph.py`: A function to convert the skeleton landmarks to a graph representation
- `utils.py`: My handy tools to select the GPU
- `datareader_gcn.py`: A custom PyTorch dataset class to read the skeleton landmarks
- `model/gcn_model.py`: The Graph Convolutional Network model for the skeleton landmarks fall detection
- `train_gcn.py`: The training script for the Graph Convolutional Network model