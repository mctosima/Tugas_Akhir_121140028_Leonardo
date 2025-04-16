import torch
from torch.optim import AdamW
from torch.nn import BCELoss
from model.gcn_improve_claude import ImprovedGCN
from kfold_dataset import FallDetectionDatasetKFold
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
from utils import check_set_gpu
import numpy as np
from sklearn.metrics import classification_report
import wandb
from datetime import datetime
import os
from k_fold_datareader import generate_kfold_splits, save_kfold_splits, verify_fold_separation

# Set dataset path
DATASET_PATH = os.path.join(os.getcwd(), 'data-skeleton')

def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_sub_epoch(model, loader, criterion, optimizer, device, fold_number=None, wandb_sync=False):
    model.train()
    total_loss = 0
    predictions = []
    labels = []
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        out = out.squeeze(1)  # Make output shape match target
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        batch_preds = (out.detach().cpu().numpy() > 0.5).astype(float)
        predictions.extend(batch_preds)
        labels.extend(batch.y.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    metrics = classification_report(labels, predictions, output_dict=True, zero_division=0)
    
    # if wandb_sync:
    #     log_dict = {
    #         "train_loss": avg_loss,
    #         "train_accuracy": metrics['accuracy'],
    #         "train_precision": metrics['weighted avg']['precision'],
    #         "train_recall": metrics['weighted avg']['recall'],
    #         "train_f1": metrics['weighted avg']['f1-score'],
    #         "epoch": epoch+1
    #     }
    #     if fold_number is not None:
    #         log_dict[f"fold{fold_number}_train_loss"] = avg_loss
    #         log_dict[f"fold{fold_number}_train_accuracy"] = metrics['accuracy']
        
    #     wandb.log(log_dict)
        
    
    return avg_loss, metrics

def validate_sub_epoch(model, loader, criterion, device, fold_number=None, wandb_sync=False):
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            out = out.squeeze(1)  # Make output shape match target
            loss = criterion(out, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            batch_preds = (out.cpu().numpy() > 0.5).astype(float)
            predictions.extend(batch_preds)
            labels.extend(batch.y.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    metrics = classification_report(labels, predictions, output_dict=True, zero_division=0)
    
    # if wandb_sync:
    #     log_dict = {
    #         "val_loss": avg_loss,
    #         "val_accuracy": metrics['accuracy'],
    #         "val_precision": metrics['weighted avg']['precision'],
    #         "val_recall": metrics['weighted avg']['recall'],
    #         "val_f1": metrics['weighted avg']['f1-score'],
    #         "epoch": epoch+1,
    #     }
        
    #     if fold_number is not None:
    #         log_dict[f"fold{fold_number}_val_loss"] = avg_loss
    #         log_dict[f"fold{fold_number}_val_accuracy"] = metrics['accuracy']
        
    #     wandb.log(log_dict)
    
    return avg_loss, metrics

def save_model(model, run_name, epoch, val_score, save_dir='pth'):
    """Save model and return the saved path"""
    save_dir = os.path.join(os.getcwd(), save_dir)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{run_name}_epoch{epoch}_val{val_score:.4f}.pth")
    torch.save(model.state_dict(), path)
    return path

def manage_saved_models(new_path, top_model_paths, max_saved=3):
    """Manage the saved model files, keeping only the top performing ones"""
    top_model_paths.append(new_path)
    # Sort paths by validation score (extract score from filename)
    sorted_paths = sorted(top_model_paths, 
                         key=lambda x: float(x.split('val')[-1].split('.pth')[0]),
                         reverse=True)
    
    # Keep only the top max_saved models
    for path in sorted_paths[max_saved:]:
        if os.path.exists(path):
            os.remove(path)
    
    return sorted_paths[:max_saved]

def train_fold(model, criterion, optimizer, fold_number, config, device, train_loaders, val_loaders, fold_train_metrics, fold_val_metrics, wandb_sync=False):
    """Train a model for a specific fold in one epoch"""
    set_seeds(config['seed'])
    
    # Load dataset for this fold (no need to load repeatedly per epoch)
    train_loader = train_loaders[fold_number - 1]  # Access the pre-loaded fold data
    val_loader = val_loaders[fold_number - 1]      # Access the pre-loaded fold data
    
    # Train and validate for each fold in this epoch
    print(f"\nFold {fold_number}")
    fold_train_loss, fold_train_metric = train_sub_epoch(model, train_loader, criterion, optimizer, device, fold_number, wandb_sync)
    fold_val_loss, fold_val_metric = validate_sub_epoch(model, val_loader, criterion, device, fold_number, wandb_sync)
    
    fold_train_metrics.append(fold_train_metric)
    fold_val_metrics.append(fold_val_metric)

    # Print the results for each fold
    print(f"Fold {fold_number} Train\nLoss: {fold_train_loss:.4f},\nAccuracy: {fold_train_metric['accuracy']:.4f},\nPrecision: {fold_train_metric['weighted avg']['precision']:.4f},\nRecall: {fold_train_metric['weighted avg']['recall']:.4f},\nF1: {fold_train_metric['weighted avg']['f1-score']:.4f}")
    print(f"\nFold {fold_number} Validation\nLoss: {fold_val_loss:.4f},\nAccuracy: {fold_val_metric['accuracy']:.4f},\nPrecision: {fold_val_metric['weighted avg']['precision']:.4f}\nRecall: {fold_val_metric['weighted avg']['recall']:.4f},\nF1: {fold_val_metric['weighted avg']['f1-score']:.4f}")

    return

def main(wandb_sync=False):
    """Main function to run 5-fold cross-validation training"""
    set_seeds(42)
    
    config = {
        'epochs': 10,
        'batch_size': 32,
        'learning_rate': 5e-4,
        'weight_decay': 1e-5,
        'num_node_features': 3,
        'hidden_channels': [64, 128, 256, 128],
        'dropout_rate': 0.1,
        'residual': True,
        'scheduler_step_size': 20,
        'scheduler_gamma': 0.5,
        'seed': 42,
        'run_name': f"5fold_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    }
    
    device = check_set_gpu()
    
    split_dir = 'splits'
    os.makedirs(split_dir, exist_ok=True)
    
    all_folds_exist = all(
        os.path.exists(os.path.join(split_dir, f'5fold_split_fold{fold}.json')) 
        for fold in range(1, 6)
    )
    
    if not all_folds_exist:
        print("Generating 5-fold splits...")
        fold_splits = generate_kfold_splits(DATASET_PATH, k=5, seed=config['seed'])
        save_kfold_splits(fold_splits)
        is_valid = verify_fold_separation(fold_splits)
        
        if is_valid:
            print("All folds have proper separation between train and validation sets!")
        else:
            print("WARNING: Some folds have overlapping train and validation files.")
    
    if wandb_sync:
        wandb.init(
            entity="mctosima",
            project="fall-detection-code-slayer",
            name=config['run_name'],
            config=config
        )


    # Initialize model
    model = ImprovedGCN(
        num_node_features=config['num_node_features'], 
        hidden_channels=config['hidden_channels'],
        dropout_rate=config['dropout_rate'],
        residual=config['residual'],
        seed=config['seed']
    )
    
    model = model.to(device)
    criterion = BCELoss()
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = StepLR(optimizer, step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'])

    # Initialize best metric trackers
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    best_accuracy_epoch = 0
    best_precision_epoch = 0
    best_recall_epoch = 0

    # Initialize list to store paths of top models
    top_model_paths = []

    # Load datasets (one time)
    train_loaders = []
    val_loaders = []
    for fold in range(1, 6):
        train_dataset = FallDetectionDatasetKFold(fold_number=fold, is_train=True)
        val_dataset = FallDetectionDatasetKFold(fold_number=fold, is_train=False)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    # Train each fold in all epochs
    for epoch in range(config['epochs']):
        # Training for 1 epoch
        fold_train_metrics = []
        fold_val_metrics = []
        print(f"\n{'='*20} Epoch {epoch+1} {'='*20}\n")
        for fold in range(1, 6):
            train_fold(model, criterion, optimizer, fold, config, device, train_loaders, val_loaders, fold_train_metrics, fold_val_metrics, wandb_sync)
        
        # Log the average metrics for this epoch and Update best metrics
        avg_train_accuracy = np.mean([metric['accuracy'] for metric in fold_train_metrics])
        avg_val_accuracy = np.mean([metric['accuracy'] for metric in fold_val_metrics])
        avg_val_precision = np.mean([metric['weighted avg']['precision'] for metric in fold_val_metrics])
        avg_val_recall = np.mean([metric['weighted avg']['recall'] for metric in fold_val_metrics])
        
        scheduler.step()

        # Save model and manage saved files
        model_path = save_model(model, config['run_name'], epoch + 1, avg_val_accuracy)
        top_model_paths = manage_saved_models(model_path, top_model_paths)
        
        print(f"\nEpoch Average Training Accuracy: {avg_train_accuracy:.4f}")
        print(f"Epoch Average Validation Accuracy: {avg_val_accuracy:.4f}")
        
        if avg_val_accuracy > best_accuracy:
            best_accuracy = avg_val_accuracy
            best_accuracy_epoch = epoch + 1
            
        if avg_val_precision > best_precision:
            best_precision = avg_val_precision
            best_precision_epoch = epoch + 1
            
        if avg_val_recall > best_recall:
            best_recall = avg_val_recall
            best_recall_epoch = epoch + 1
    
    
    # Print best metrics
    print("\nBest Metrics Summary:")
    print(f"Best Accuracy: {best_accuracy:.4f} (Epoch {best_accuracy_epoch})")
    print(f"Best Precision: {best_precision:.4f} (Epoch {best_precision_epoch})")
    print(f"Best Recall: {best_recall:.4f} (Epoch {best_recall_epoch})")

    # Print final saved models
    print("\nFinal Saved Models:")
    for path in top_model_paths:
        print(f"- {path}")
    
    if wandb_sync:
        wandb.finish()

if __name__ == '__main__':
    main(wandb_sync=False)