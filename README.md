# Fall Detection using Graph Convolutional Network Based on Skeleton Landmarks Mediapipe

## Getting Started
1. Clone this repository
2. add 2 new folders: data-skeleton and skeleton-preview 
2. Create a virtual environment with Python 3.10. Activate the virtual environment
3. Install the required packages
    ```bash
    pip install -r requirements.txt
    ```
---

## How to Run the Project

To execute this project, please follow the steps outlined below.

### 1. Environment Setup

Ensure that a virtual environment has been created and activated before running any scripts. All required dependencies should also be installed.

### 2. Frame Extraction

1. Download the dataset that can be accessed with this [link](http://kuleuven.box.com/s/dyo66et36l2lqvl19i9i7p66761sy0s6) and place it inside the `extract_frame_program` directory.

2. Navigate to the directory:

   ```bash
   cd extract_frame_program
   ```

3. Execute the frame extraction script:

   ```bash
   python frame_extract_random.py
   ```

4. The extracted frames will be stored in the `extracted_frames` directory.

### 3. Skeleton Extraction

1. Move the `extracted_frames` directory to the main project workspace.

2. Navigate back to the root workspace:

   ```bash
   cd ..
   ```

3. Run the skeleton extraction script:

   ```bash
   python extract_skeleton.py
   ```

4. This process extracts skeleton data from the frames using MediaPipe Pose and saves the output in JSON format.

### 4. Model Training

Run the following command to train the model:

```bash
python train_gcn_kfold.py
```

This script will:
* Train the model using 5-fold cross-validation
* Save the best-performing model weights in the `pth` directory for future evaluation
---

## How to testing the model

1. copy the best model weights from the `pth` folder to the `testing` folder
2. Run the following command:
```bash
python "testing/test_gcn_kfold.py"
```
---

## Project Structure

- `data-skeleton`: Contains the extracted skeleton landmarks from the dataset.
- `extract_frame_program`: Contains the program to extract the frames from the video dataset.
- `skeleton-preview`: Contains the preview of the skeleton landmarks in .jpg format.
- `splits`: Contains the train and validation splits path (implementing 5-Fold splits) in .json format
- `model`: Contains the PyTorch model.
- `pth`: Contains the trained model weights.
- `testing`: Contains the testing data (skeleton landmarks) and the testing script.
---

## `.py` Files

- `extract_frame_program\frame_extract_random.py`: A script to extract the frames from the train video dataset.
- `extract_skeleton.py`: Extract the skeleton landmarks from the dataset.
- `dataset_rebalancing.py`: Remove random skeleton data from a class to balance the totals for each class
- `func_distance_features.py`: A function to calculate the distance features from the skeleton landmarks.
- `func_lm_to_graph.py`: A function to convert the skeleton landmarks to a graph representation.
- `utils.py`: My handy tools to select the GPU
- `k_fold_datareader.py`: A custom PyTorch program that contains the functions to read the skeleton landmarks, generate 5-fold splits then save it to folder splits, and verify those 5-Fold split files.
- `k_fold_generator.py`: A custom PyTorch class that contains the functions to read the 5-Fold split files and generate the graph representation of the skeleton landmarks.
- `model/gcn_model.py`: The Graph Convolutional Network model for the skeleton landmarks fall detection.
- `train_gcn_kfold.py`: The main script for the project: including read, load dataset and train the Graph Convolutional Network model.
- `utils.py`: A custom PyTorch function to detect any appropriate gpu that can be used for faster training.
- `testing/extract_skeleton.py`: A script to extract the skeleton landmarks from the testing dataset.
- `testing/video_frame_extract.py`: A script to extract the frames from the test video dataset.
- `testing/prediction.py`: A script to predict the fall detection using the trained model.
---

## How to Cite

If you use this project, please cite:

### BibTeX
```bibtex
@thesis{sirait2026fall_detection,
  author       = {Leonardo Alfontus Mende Sirait},
  title        = {Pengembangan Deteksi Posisi Jatuh Pada Lansia Menggunakan Graph Convolutional Network Berbasis Pose Landmark},
  year         = {2026},
  school       = {Institut Teknologi Sumatera},
  type         = {Undergraduate Thesis},
  address      = {Indonesia}
}
