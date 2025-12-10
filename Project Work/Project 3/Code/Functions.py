import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import matplotlib.pyplot as plt

def load_realwaste_dataset(data_dir, img_size=(128, 128), test_size=0.2, random_state=42, grayscale=False,drop=[None]):
    """
    Load RealWaste dataset with stratified train-test split.
    
    Parameters:
    -----------
    data_dir : str
        Path to the RealWaste directory containing category folders
    img_size : tuple
        Target size for images (height, width)
    test_size : float
        Proportion of dataset to include in test split (default: 0.2)
    random_state : int
        Random seed for reproducibility
    grayscale : bool
        If True, convert images to grayscale (default: False for RGB)
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : numpy arrays
        Training and test sets
    class_names : list
        List of class names corresponding to label indices
    """
    
    # Define the class folders
    class_folders = [
        'Cardboard', 'Food Organics', 'Glass', 'Metal', 
        'Miscellaneous Trash', 'Paper', 'Plastic', 
        'Textile Trash', 'Vegetation'
    ]
    
    if drop[0] is not None:
        class_folders = [cf for cf in class_folders if cf not in drop]
    
    images = []
    labels = []
    class_names = []
    
    print("Loading images from dataset...")
    for class_idx, class_name in enumerate(class_folders):
        class_path = Path(data_dir) / class_name
        if not class_path.exists():
            print(f"Warning: {class_name} folder not found at {class_path}")
            continue
            
        class_names.append(class_name)
        image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        
        print(f"Loading {len(image_files)} images from {class_name}...")
        for img_path in image_files:
            try:
                # Load and resize image
                img = Image.open(img_path)
                
                # Convert to grayscale or RGB
                if grayscale:
                    img = img.convert('L')  # Grayscale
                else:
                    img = img.convert('RGB')  # RGB
                
                img = img.resize(img_size)
                
                # Convert to numpy array and normalize to [0, 1]
                img_array = np.array(img) / 255.0
                
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    print(f"\nDataset loaded:")
    print(f"Total images: {len(X)}")
    print(f"Image shape: {X.shape[1:]}")
    print(f"Number of classes: {len(class_names)}")
    print(f"\nClass distribution:")
    for idx, class_name in enumerate(class_names):
        count = np.sum(y == idx)
        print(f"  {class_name}: {count} images")
    
    # Stratified train-test split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")
    print(f"\nTrain set class distribution:")
    for idx, class_name in enumerate(class_names):
        count = np.sum(y_train == idx)
        print(f"  {class_name}: {count} images ({count/len(y_train)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test, class_names


import numpy as np

def augment_training_data(X_train, y_train):
    """
    Augments the training set by adding rotations and flips.
    
    Parameters:
    -----------
    X_train : numpy array
        Original training images (N, H, W, C) or (N, H, W)
    y_train : numpy array
        Original training labels (N,)
        
    Returns:
    --------
    X_aug, y_aug : numpy arrays
        The augmented training set containing original + new images
    """
    print("training set size: ", len(X_train))
    print("Starting data augmentation...")
    
    X_aug_list = [X_train]
    y_aug_list = [y_train]
    
    # 1. Flip Left-Right
    X_lr = np.flip(X_train, axis=2) # axis 2 is width
    X_aug_list.append(X_lr)
    y_aug_list.append(y_train)
    
    # 2. Flip Up-Down
    X_ud = np.flip(X_train, axis=1) # axis 1 is height
    X_aug_list.append(X_ud)
    y_aug_list.append(y_train)
    
    # 3. Rotate 90 degrees
    X_rot90 = np.rot90(X_train, k=1, axes=(1, 2)) 
    X_aug_list.append(X_rot90)
    y_aug_list.append(y_train)
    
    # 4. Rotate 180 degrees
    X_rot180 = np.rot90(X_train, k=2, axes=(1, 2))
    X_aug_list.append(X_rot180)
    y_aug_list.append(y_train)

    # 5. Rotate 270 degrees
    X_rot270 = np.rot90(X_train, k=3, axes=(1, 2))
    X_aug_list.append(X_rot270)
    y_aug_list.append(y_train)
    
    # Combine all lists
    X_aug = np.concatenate(X_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)
    
    
    print(f"Augmentation complete. Final training set size: {len(X_aug)}")
    return X_aug, y_aug