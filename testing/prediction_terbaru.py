# testing/main.py
import os
import json
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
from torch.nn import BatchNorm1d, Dropout, Linear

# ---------------------------
# ====== Model definition ===
# ---------------------------
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
    def __init__(
        self,
        num_node_features: int,
        hidden_channels: int | list,
        num_classes: int = 1,
        dropout_rate: float = 0.3,
        pool_type: str = 'mean',
        residual: bool = True,
        seed: int = 42
    ):
        super(SkeletonGCN, self).__init__()
        torch.manual_seed(seed)

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels, hidden_channels]

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.residual = residual

        # input layer
        self.convs.append(GCNConv(num_node_features, hidden_channels[0]))
        self.batch_norms.append(BatchNorm1d(hidden_channels[0]))

        # hidden
        for i in range(len(hidden_channels) - 1):
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i + 1]))
            self.batch_norms.append(BatchNorm1d(hidden_channels[i + 1]))

        # head
        self.dropout = Dropout(p=dropout_rate)
        self.final_lin1 = Linear(hidden_channels[-1], max(1, hidden_channels[-1] // 2))
        self.final_lin2 = Linear(max(1, hidden_channels[-1] // 2), num_classes)

        # pooling
        if pool_type == 'max':
            self.pool = global_max_pool
        elif pool_type == 'add':
            self.pool = global_add_pool
        else:
            self.pool = global_mean_pool

    def forward(self, x, edge_index, batch):
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

        # If binary (out_features == 1) -> sigmoid, else log_softmax
        if self.final_lin2.out_features == 1:
            return torch.sigmoid(x)
        else:
            return F.log_softmax(x, dim=-1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()
        self.final_lin1.reset_parameters()
        self.final_lin2.reset_parameters()


# ---------------------------
# ====== Utilities ==========
# ---------------------------
LANDMARK_NAMES = [
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

def make_undirected_edge_index(edges: List[Tuple[int,int]]) -> torch.LongTensor:
    """Convert list of (u,v) to undirected edge_index tensor [2, num_edges*2]"""
    all_edges = []
    for u, v in edges:
        all_edges.append((u, v))
        all_edges.append((v, u))
    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    return edge_index

UNDIRECTED_EDGE_INDEX = make_undirected_edge_index(skeleton_edges)

def landmarks_to_graph(landmarks: Dict, edge_index: torch.LongTensor) -> Data:
    node_features = []
    for name in LANDMARK_NAMES:
        landmark = landmarks.get(name, {"x": 0.0, "y": 0.0, "z": 0.0})
        node_features.append([landmark.get('x', 0.0), landmark.get('y', 0.0), landmark.get('z', 0.0)])
    x = torch.tensor(node_features, dtype=torch.float)   # [num_nodes, 3]
    data = Data(x=x, edge_index=edge_index)
    return data

def predict_skeleton_action(model: nn.Module, skeleton_graph: Data, device='cpu'):
    """
    Returns:
      predicted_label_str, predicted_prob (float), raw_output_tensor
    """
    model.eval()
    skeleton_graph.x = skeleton_graph.x.to(device)
    skeleton_graph.edge_index = skeleton_graph.edge_index.to(device)
    # create batch vector for single graph: all zeros
    batch = torch.zeros(skeleton_graph.x.size(0), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(skeleton_graph.x, skeleton_graph.edge_index, batch)  # shape [1, num_classes] typically
        out = out.cpu()
        if out.size(-1) == 1:
            prob = float(out.squeeze().item())  # sigmoid probability
            pred_idx = 1 if prob >= 0.5 else 0
            return pred_idx, prob, out
        else:
            # multi-class: out is log_softmax (log-probs)
            probs = torch.exp(out)  # convert log-probs to probs
            prob_val, pred_idx = torch.max(probs, dim=-1)
            return int(pred_idx.item()), float(prob_val.item()), out

# ---------------------------
# ====== Main prediction ====
# ---------------------------
def process_skeletons_for_prediction(skeleton_dir_fall: str, skeleton_dir_not_fall: str, model_path: str, output_csv: str, device='cpu'):
    # --- instantiate model architecture (adjust hidden_channels & num_classes to match your trained model)
    # NOTE: change hidden_channels and num_classes if your trained model used different config
    model = SkeletonGCN(
        num_node_features=3,
        hidden_channels=[64, 32, 32, 32, 32],
        num_classes=1,   # set to >1 if your model is multi-class
        dropout_rate=0.3,
        residual=True,
        seed=42
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # # load state dict safely to CPU/GPU as required
    # state = torch.load(model_path, map_location=device)
    # # if user saved a dict like {'model_state_dict': ...}, try detect
    # if isinstance(state, dict) and 'state_dict' in state:
    #     state = state['state_dict']
    # # Allow both direct state_dict or wrapped dict
    # if isinstance(state, dict) and any(k.startswith('module.') for k in state.keys()):
    #     # if saved from DataParallel, strip "module."
    #     new_state = {}
    #     for k, v in state.items():
    #         new_state[k.replace('module.', '')] = v
    #     state = new_state
    
    # model.load_state_dict(state)
    state = torch.load(model_path, map_location=device)
    print(f"[DEBUG] loaded checkpoint type: {type(state)}")

    # If checkpoint is a dict, try to extract model-state dict using common keys
    if isinstance(state, dict):
        # print top-level keys for debugging
        print("[DEBUG] checkpoint keys:", list(state.keys()))

        # candidate keys that may contain the model weights
        candidate_model_keys = ['state_dict', 'model_state', 'model_state_dict', 'model', 'model_state_dict_ema']
        model_state = None
        for k in candidate_model_keys:
            if k in state:
                model_state = state[k]
                print(f"[DEBUG] using checkpoint['{k}'] as model_state")
                break

        # If still None, maybe entire dict *is* the state_dict (some pipelines save directly)
        if model_state is None:
            # Heuristic: if values in dict are tensors, it's likely a state_dict
            some_vals = list(state.values())[:5]
            if len(some_vals) > 0 and all(hasattr(v, 'shape') for v in some_vals):
                model_state = state
                print("[DEBUG] checkpoint looks like a raw state_dict; using it directly.")
            else:
                raise RuntimeError(f"Cannot find model weights in checkpoint. Keys: {list(state.keys())}")

    else:
        # Not a dict: assume it's already a state_dict-like object (rare)
        model_state = state

    # If loaded state has DataParallel 'module.' prefixes, strip them
    if isinstance(model_state, dict):
        if any(k.startswith('module.') for k in model_state.keys()):
            new_state = {}
            for k, v in model_state.items():
                new_state[k.replace('module.', '')] = v
            model_state = new_state
            print("[DEBUG] Stripped 'module.' prefix from state_dict keys (DataParallel).")

    # Before loading, optional: compare keys and show helpful diffs (useful to detect architecture mismatch)
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(model_state.keys())
    epoch = state.get('epoch', 'N/A') if isinstance(state, dict) else 'N/A'
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    print(f"[INFO] Loading model from {model_path} (epoch: {epoch})")
    print(f"[DEBUG] model keys count: {len(model_keys)}, ckpt keys count: {len(ckpt_keys)}")
    print(f"[DEBUG] missing keys (in model but not in ckpt): {list(missing)[:10]}{'...' if len(missing)>10 else ''}")
    print(f"[DEBUG] unexpected keys (in ckpt but not in model): {list(unexpected)[:10]}{'...' if len(unexpected)>10 else ''}")

    # Try to load, allowing non-strict if necessary (but print warning)
    try:
        model.load_state_dict(model_state, strict=True)
        print("[INFO] state_dict loaded with strict=True")
    except RuntimeError as e:
        print("[WARN] strict=True failed:", e)
        print("[INFO] trying to load with strict=False (will ignore missing/unexpected keys)")
        model.load_state_dict(model_state, strict=False)
        print("[INFO] state_dict loaded with strict=False")

    model.eval()

    results = []
    # helper to process one directory and attach true label
    def process_dir(dir_path: str, true_label_str: str):
        if not os.path.exists(dir_path):
            print(f"[WARNING] directory not found: {dir_path}. Skipping.")
            return []
        files = sorted([f for f in os.listdir(dir_path) if f.endswith('.json')])
        out_rows = []
        for fname in files:
            fpath = os.path.join(dir_path, fname)
            with open(fpath, 'r') as fh:
                data = json.load(fh)
            graph = landmarks_to_graph(data, UNDIRECTED_EDGE_INDEX)
            pred_idx, prob, raw_out = predict_skeleton_action(model, graph, device=device)
            # Map numeric pred_idx to label string. ASSUMPTION: 1 -> 'fall', 0 -> 'not_fall'
            # If your model uses other mapping, change below mapping dict.
            idx_to_label = {0: 'not_fall', 1: 'fall'}
            pred_label = idx_to_label.get(pred_idx, str(pred_idx))
            out_rows.append({
                'filename': fname,
                'true_label': true_label_str,
                'predicted_label': pred_label,
                'predicted_prob': prob
            })
        return out_rows

    results += process_dir(skeleton_dir_fall, 'fall')
    results += process_dir(skeleton_dir_not_fall, 'not_fall')

    # Save CSV
    df = pd.DataFrame(results, columns=['filename', 'true_label', 'predicted_label', 'predicted_prob'])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Predictions saved to {output_csv}")

    # Confusion matrix (requires at least one sample per class)
    if len(df) == 0:
        print("[WARN] no predictions were produced (empty dataset).")
        return df

    # convert labels to integers for confusion matrix
    label_to_idx = {'fall': 1, 'not_fall': 0}
    y_true = df['true_label'].map(label_to_idx).astype(int).values
    y_pred = df['predicted_label'].map(label_to_idx).fillna(-1).astype(int).values

    # If any predicted labels are outside mapping (fillna -> -1), filter them but warn
    if (y_pred == -1).any():
        print("[WARN] some predicted labels were not in label_to_idx mapping; they are excluded from confusion matrix.")
        mask = (y_pred != -1)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,labels=[1, 0], target_names=['fall', 'not_fall'], zero_division=0))

    # plot and save confusion matrix figure
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ['fall', 'not_fall']
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix')
    # annotate cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    cm_img_path = os.path.splitext(output_csv)[0] + "_confusion_matrix.png"
    fig.savefig(cm_img_path)
    print(f"[INFO] Confusion matrix image saved to: {cm_img_path}")

    return df


# ---------------------------
# ====== Run as script ======
# ---------------------------
if __name__ == '__main__':
    # Adjust these paths if necessary
    base_dir = os.path.join(os.getcwd(), 'testing')
    model_filename = 'best.pth'
    model_path = os.path.join(base_dir, model_filename)

    skeleton_dir_fall = os.path.join(base_dir, 'data-skeleton', 'fall')
    skeleton_dir_not_fall = os.path.join(base_dir, 'data-skeleton', 'not_fall')

    output_csv = os.path.join(base_dir, 'predictions_all.csv')

    # Device: use CPU by default; change to 'cuda' if available and model saved for GPU
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print("[INFO] CUDA available. Using GPU.")

    df = process_skeletons_for_prediction(skeleton_dir_fall, skeleton_dir_not_fall, model_path, output_csv, device=device)
