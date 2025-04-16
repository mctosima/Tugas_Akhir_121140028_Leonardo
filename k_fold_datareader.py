import numpy as np
import json
import os
from glob import glob
from sklearn.model_selection import KFold
import torch

def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seeds()

def save_kfold_splits(fold_splits, base_save_dir='splits'):
    """
    Save k-fold cross validation splits to JSON files.
    
    Args:
        fold_splits (list): List of dictionaries containing train and val files for each fold
        base_save_dir (str): Directory to save the split files
    """
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Convert absolute paths to relative for storage
    for i, split in enumerate(fold_splits):
        # Convert absolute paths to relative paths
        rel_train_files = [os.path.relpath(f, os.getcwd()) for f in split['train_files']]
        rel_val_files = [os.path.relpath(f, os.getcwd()) for f in split['val_files']]
        
        # Create data structure for this fold
        fold_data = {
            'fold': i + 1,
            'split_mode': '5-Fold',
            'train_files': rel_train_files,
            'val_files': rel_val_files
        }
        
        # Save to JSON file
        save_path = os.path.join(base_save_dir, f'5fold_split_fold{i+1}.json')
        with open(save_path, 'w') as f:
            json.dump(fold_data, f, indent=4)
        
        print(f"Saved fold {i+1} to {save_path}")

def generate_kfold_splits(dataset_path, k=5, seed=42):
    """
    Generate k-fold cross-validation splits for the dataset.
    This ensures that files from the same subject/recording stay together in the same split.
    
    Args:
        dataset_path (str): Path to the dataset root
        k (int): Number of folds
        seed (int): Random seed for reproducibility
        
    Returns:
        list: List of dictionaries containing train and val files for each fold
    """
    set_seeds(seed)
    
    fall_category = 'fall'
    not_fall_category = 'not_fall'
    
    # Get all group prefixes from the folder names
    all_prefixes = set()
    
    # Get all fall folders
    fall_folders = os.listdir(os.path.join(dataset_path, fall_category))
    for folder in fall_folders:
        prefix = folder.split('_')[0]
        all_prefixes.add(prefix)
    
    # Get all not_fall folders
    not_fall_folders = os.listdir(os.path.join(dataset_path, not_fall_category))
    for folder in not_fall_folders:
        prefix = folder.split('_')[0]
        all_prefixes.add(prefix)
    
    # Convert to sorted list for deterministic behavior
    all_prefixes = sorted(list(all_prefixes))
    
    print(f"Found {len(all_prefixes)} unique prefixes: {all_prefixes}")
    
    # Create a mapping from prefix to all files in that group
    prefix_to_files = {}
    
    # Process fall files
    for prefix in all_prefixes:
        matching_fall_folders = [folder for folder in fall_folders if folder.split('_')[0] == prefix]
        matching_not_fall_folders = [folder for folder in not_fall_folders if folder.split('_')[0] == prefix]
        
        all_files = []
        
        # Get all fall files for this prefix
        for folder in matching_fall_folders:
            folder_path = os.path.join(dataset_path, fall_category, folder)
            json_files = glob(os.path.join(folder_path, "*.json"))
            all_files.extend(json_files)
        
        # Get all not_fall files for this prefix
        for folder in matching_not_fall_folders:
            folder_path = os.path.join(dataset_path, not_fall_category, folder)
            json_files = glob(os.path.join(folder_path, "*.json"))
            all_files.extend(json_files)
        
        prefix_to_files[prefix] = all_files
        print(f"Prefix {prefix}: {len(all_files)} files")
    
    # Initialize k-fold cross-validation on the prefixes, not individual files
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    
    # Generate splits based on prefixes
    fold_splits = []
    
    # Convert prefixes to list for sklearn's KFold
    prefix_list = list(all_prefixes)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(prefix_list)):
        train_prefixes = [prefix_list[i] for i in train_idx]
        val_prefixes = [prefix_list[i] for i in val_idx]
        
        train_files = []
        val_files = []
        
        # Collect all files for train and val splits based on their prefixes
        for prefix in train_prefixes:
            train_files.extend(prefix_to_files[prefix])
        
        for prefix in val_prefixes:
            val_files.extend(prefix_to_files[prefix])
        
        # Check for any overlap between train and validation files
        train_set = set(train_files)
        val_set = set(val_files)
        overlap = train_set.intersection(val_set)
        if overlap:
            print(f"WARNING: Found {len(overlap)} overlapping files between train and validation!")
            for file in overlap:
                print(f"  Overlapping file: {file}")
        
        # Add split to result
        fold_splits.append({
            'fold': fold + 1,
            'train_prefixes': train_prefixes,
            'val_prefixes': val_prefixes,
            'train_files': train_files,
            'val_files': val_files
        })
        
        print(f"Fold {fold+1}:")
        print(f"  Train: {len(train_prefixes)} prefixes, {len(train_files)} files")
        print(f"  Validation: {len(val_prefixes)} prefixes, {len(val_files)} files")
    
    return fold_splits

def load_fold_split(fold_number, base_split_dir='splits'):
    """
    Load a specific fold split from a JSON file
    
    Args:
        fold_number (int): Which fold to load (1-based index)
        base_split_dir (str): Directory containing the split files
        
    Returns:
        dict: Dictionary with train_files and val_files
    """
    split_file = os.path.join(base_split_dir, f'5fold_split_fold{fold_number}.json')
    
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file {split_file} not found. Please run generate_kfold_splits first.")
    
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    # Convert relative paths to absolute
    train_files = [os.path.join(os.getcwd(), f) for f in split_data['train_files']]
    val_files = [os.path.join(os.getcwd(), f) for f in split_data['val_files']]
    
    return {
        'train_files': train_files,
        'val_files': val_files
    }

def verify_fold_separation(fold_splits):
    """
    Verify that there is no overlap between train and validation sets
    in each fold.
    
    Args:
        fold_splits (list): List of dictionaries containing train and val files for each fold
    
    Returns:
        bool: True if there's no overlap, False otherwise
    """
    all_valid = True
    
    for i, split in enumerate(fold_splits):
        train_files_set = set(split['train_files'])
        val_files_set = set(split['val_files'])
        
        intersection = train_files_set.intersection(val_files_set)
        
        if intersection:
            print(f"ERROR: Fold {i+1} has {len(intersection)} overlapping files!")
            print(f"Example overlapping files: {list(intersection)[:5]}")
            all_valid = False
        else:
            print(f"Fold {i+1}: No overlap between train and validation sets ✓")
    
    return all_valid

if __name__ == '__main__':
    # Set the path to your dataset
    DATASET_PATH = os.path.join(os.getcwd(), 'data-skeleton')
    
    # Generate k-fold splits
    fold_splits = generate_kfold_splits(DATASET_PATH, k=5, seed=42)
    
    # Verify no overlap between train and validation for each fold
    is_valid = verify_fold_separation(fold_splits)
    
    if is_valid:
        print("All folds have proper separation between train and validation sets!")
        # Save the splits to files
        save_kfold_splits(fold_splits)
    else:
        print("WARNING: Some folds have overlapping train and validation files.")
        print("Please fix the issues before saving splits.")
