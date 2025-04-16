import torch
from torch_geometric.data import Data
from func_distance_feature import extract_engineered_features


# Define the skeleton connections based on MediaPipe's 33 landmarks
# Each tuple represents a connection between two landmarks (source, target)
# Landmark indices based on MediaPipe documentation

skeleton_edges = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 13), (13, 15), (15, 17),
    (15, 19), (15, 21), (17, 19), (19, 21),
    (11, 23), (23, 25), (25, 27), (27, 29), (27, 31),
    (23, 24), (24, 26), (26, 28), (28, 30), (28, 32)
    # Add more connections as needed based on the full 33 landmarks
]

def landmarks_to_graph(landmarks, label, skeleton_edges):
    # Extract node features (x, y, z)
    node_features = []
    landmark_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_pinky", "right_pinky", "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]
    
    for name in landmark_names:
        landmark = landmarks.get(name, {"x": 0.0, "y": 0.0, "z": 0.0})
        node_features.append([landmark['x'], landmark['y'], landmark['z']])
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Define edge_index based on skeleton_edges
    edge_index = torch.tensor(skeleton_edges, dtype=torch.long).t().contiguous()
    
    # Label
    y = torch.tensor([label], dtype=torch.float)
    
    # Create graph data
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data

def landmarks_to_graph_with_features(landmarks, label, skeleton_edges):
    # Extract node features (x, y, z)
    node_features = []
    landmark_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_pinky", "right_pinky", "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]
    
    for name in landmark_names:
        landmark = landmarks.get(name, {"x": 0.0, "y": 0.0, "z": 0.0})
        node_features.append([landmark['x'], landmark['y'], landmark['z']])
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Define edge_index based on skeleton_edges
    edge_index = torch.tensor(skeleton_edges, dtype=torch.long).t().contiguous()
    
    # Label
    y = torch.tensor([label], dtype=torch.float)
    
    # Engineered Features
    engineered = extract_engineered_features(landmarks)
    engineered_features = torch.tensor([list(engineered.values())], dtype=torch.float)
    
    # Attach engineered features as global attributes
    data = Data(x=x, edge_index=edge_index, y=y, engineered=engineered_features)
    
    
    return data

