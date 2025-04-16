import os
import json
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
from torch.nn import BatchNorm1d, Dropout, Linear
from torch_geometric.loader import DataLoader

# Definisikan struktur model sesuai dengan model yang Anda latih
skeleton_edges = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 13), (13, 15), (15, 17),
    (15, 19), (15, 21), (17, 19), (19, 21),
    (11, 23), (23, 25), (25, 27), (27, 29), (27, 31),
    (23, 24), (24, 26), (26, 28), (28, 30), (28, 32)
]

class SkeletonGCN(nn.Module):
    def __init__(self, num_node_features: int, hidden_channels: int | list, num_classes: int = 1, dropout_rate: float = 0.3, pool_type: str = 'mean', residual: bool = True, seed: int = 42):
        super(SkeletonGCN, self).__init__()
        torch.manual_seed(seed)
        
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels, hidden_channels]
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.residual = residual
        
        self.convs.append(GCNConv(num_node_features, hidden_channels[0]))
        self.batch_norms.append(BatchNorm1d(hidden_channels[0]))
        
        for i in range(len(hidden_channels) - 1):
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i + 1]))
            self.batch_norms.append(BatchNorm1d(hidden_channels[i + 1]))
        
        self.dropout = Dropout(p=dropout_rate)
        self.final_lin1 = Linear(hidden_channels[-1], hidden_channels[-1] // 2)
        self.final_lin2 = Linear(hidden_channels[-1] // 2, num_classes)
        
        if pool_type == 'max':
            self.pool = global_max_pool
        elif pool_type == 'add':
            self.pool = global_add_pool
        else:
            self.pool = global_mean_pool

    def forward(self, x, edge_index, batch=None):
        previous = None
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            if i > 0 and self.residual:
                previous = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            if i > 0 and self.residual and x.size(-1) == previous.size(-1):
                x = x + previous
        
        x = self.pool(x, batch)
        x = self.dropout(x)
        x = F.relu(self.final_lin1(x))
        x = self.dropout(x)
        x = self.final_lin2(x)
        
        if self.final_lin2.out_features == 1:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=-1)

def landmarks_to_graph(landmarks, skeleton_edges):
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
    edge_index = torch.tensor(skeleton_edges, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    
    return data

def predict_skeleton_action(model, skeleton_graph):
    model.eval()
    with torch.no_grad():
        output = model(skeleton_graph.x, skeleton_graph.edge_index)  # Direct graph input
        predicted_class = torch.argmax(output).item()
    return predicted_class

def process_skeletons_for_prediction(skeleton_dir, model_path):
    model = SkeletonGCN(
        num_node_features=3, 
        hidden_channels=[64, 128, 256, 128],
        dropout_rate=0.1,
        residual=True,
        seed=42
    )
    
    model_path = os.path.join(os.getcwd(), 'test', model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    model.load_state_dict(torch.load(model_path, weights_only=True))  # Corrected for security and performance
    model.eval()
    
    skeleton_files = sorted([f for f in os.listdir(skeleton_dir) if f.endswith('.json')])
    
    predictions = {}
    for skeleton_file in skeleton_files:
        skeleton_path = os.path.join(skeleton_dir, skeleton_file)
        
        with open(skeleton_path, 'r') as f:
            skeleton_data = json.load(f)
        
        skeleton_graph = landmarks_to_graph(skeleton_data, skeleton_edges)
        
        prediction = predict_skeleton_action(model, skeleton_graph)
        predictions[skeleton_file] = prediction
    
    print("Prediksi Skeleton:")
    for file, pred in predictions.items():
        print(f"{file}: {pred}")
    
    return predictions

# Example usage
if __name__ == '__main__':
    model_path = '5fold_20250414171717_epoch3_val1.0000.pth'
    skeleton_dir_fall = os.path.join(os.getcwd(), 'test', 'data-skeleton', 'fall')
    skeleton_dir_not_fall = os.path.join(os.getcwd(), 'test', 'data-skeleton', 'not_fall')
    
    if not os.path.exists(skeleton_dir_fall):
        raise FileNotFoundError(f"Skeleton directory {skeleton_dir_fall} does not exist.")
    
    predictions_fall = process_skeletons_for_prediction(skeleton_dir_fall, model_path)
    predictions_not_fall = process_skeletons_for_prediction(skeleton_dir_not_fall, model_path)
