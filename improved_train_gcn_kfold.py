import torch
from torch.optim import AdamW
from torch.nn import BCELoss
from model.gcn_model import ImprovedGCN
from kfold_dataset import FallDetectionDatasetKFold
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
from utils import check_set_gpu
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
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

def train_epoch(model, loader, criterion, optimizer, device, wandb_sync=False):
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
    
    if wandb_sync:
        log_dict = {
            "train_loss": avg_loss,
            "train_accuracy": metrics['accuracy'],
            "train_precision": metrics['weighted avg']['precision'],
            "train_recall": metrics['weighted avg']['recall'],
            "train_f1": metrics['weighted avg']['f1-score']
        }
        wandb.log(log_dict)
    
    return avg_loss, metrics

def validate_epoch(model, loader, criterion, device, wandb_sync=False):
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
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    
    if wandb_sync:
        log_dict = {
            "val_loss": avg_loss,
            "val_accuracy": metrics['accuracy'],
            "val_precision": metrics['weighted avg']['precision'],
            "val_recall": metrics['weighted avg']['recall'],
            "val_f1": metrics['weighted avg']['f1-score']
        }
        wandb.log(log_dict)
    
    return avg_loss, metrics, cm

def save_model(model, run_name, fold, epoch, val_score, save_dir='pth'):
    """Save model and return the saved path"""
    save_dir = os.path.join(os.getcwd(), save_dir)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{run_name}_fold{fold}_epoch{epoch}_val{val_score:.4f}.pth")
    torch.save(model.state_dict(), path)
    return path

def manage_saved_models(new_path, top_model_paths, max_saved=1):
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

def train_single_fold(fold_number, config, device, wandb_sync=False):
    """Train a model for a specific fold"""
    set_seeds(config['seed'])
    
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
    
    # Load dataset for this fold
    train_dataset = FallDetectionDatasetKFold(fold_number=fold_number, is_train=True)
    val_dataset = FallDetectionDatasetKFold(fold_number=fold_number, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print(f"\n{'='*20} Training Fold {fold_number} {'='*20}")
    
    # Initialize best metrics and saved models list for this fold
    best_val_accuracy = 0
    best_val_epoch = 0
    top_model_paths = []
    final_confusion_matrix = None
    
    # Training loop for this fold
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train and validate
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, wandb_sync)
        val_loss, val_metrics, val_cm = validate_epoch(model, val_loader, criterion, device, wandb_sync)
        
        # Save results from last epoch
        if epoch == config['epochs'] - 1:
            final_confusion_matrix = val_cm
            
        # Update best metrics
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            best_val_epoch = epoch + 1
            
        # Save model
        model_path = save_model(model, config['run_name'], fold_number, epoch + 1, val_metrics['accuracy'])
        top_model_paths = manage_saved_models(model_path, top_model_paths)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Precision: {val_metrics['weighted avg']['precision']:.4f}, " +
              f"Recall: {val_metrics['weighted avg']['recall']:.4f}, " +
              f"F1: {val_metrics['weighted avg']['f1-score']:.4f}")
        
        # Update learning rate
        scheduler.step()
    
    # Print final results for this fold
    print(f"\nFold {fold_number} Best Validation Accuracy: {best_val_accuracy:.4f} (Epoch {best_val_epoch})")
    print(f"Final Confusion Matrix for Fold {fold_number}:")
    print(final_confusion_matrix)
    
    return {
        'fold': fold_number,
        'best_accuracy': best_val_accuracy,
        'best_epoch': best_val_epoch,
        'confusion_matrix': final_confusion_matrix,
        'top_models': top_model_paths
    }

def main(wandb_sync=False):
    """Main function to run proper 5-fold cross-validation training"""
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
    
    # Ensure KFold splits exist
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
            return
    
    if wandb_sync:
        wandb.init(
            entity="mctosima",
            project="fall-detection-code-slayer",
            name=config['run_name'],
            config=config
        )
    
    # Train each fold independently
    fold_results = []
    all_confusion_matrices = []
    
    for fold in range(1, 6):
        # Train a separate model for each fold
        fold_result = train_single_fold(fold, config, device, wandb_sync)
        fold_results.append(fold_result)
        all_confusion_matrices.append(fold_result['confusion_matrix'])
    
    # Average confusion matrix across all folds
    avg_confusion_matrix = np.mean(all_confusion_matrices, axis=0)
    
    # Print summary of all folds
    print("\n" + "="*50)
    print("CROSS-VALIDATION SUMMARY")
    print("="*50)
    
    accuracies = [result['best_accuracy'] for result in fold_results]
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    print(f"Mean Accuracy across folds: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print("\nBest Results by Fold:")
    for result in fold_results:
        print(f"Fold {result['fold']}: Accuracy = {result['best_accuracy']:.4f} (Epoch {result['best_epoch']})")
    
    print("\nAverage Confusion Matrix:")
    print(avg_confusion_matrix)
    
    # Print best model paths for each fold
    print("\nBest Model Paths:")
    for result in fold_results:
        print(f"Fold {result['fold']}:")
        for path in result['top_models']:
            print(f"- {path}")
    
    if wandb_sync:
        # Log final cross-validation results
        wandb.log({
            "cv_mean_accuracy": mean_accuracy,
            "cv_std_accuracy": std_accuracy
        })
        wandb.finish()

if __name__ == '__main__':
    main(wandb_sync=False)