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

        # Utilizing the sorted function to ensure consistent order
        image_files = sorted(list(class_path.glob('*.jpg')) + list(class_path.glob('*.png')))
        
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

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

def build_ffnn_model_PCA_optimal(input_shape, num_classes):
    """
    Builds a Feed-Forward Neural Network (Dense layers only).
    """
    model = models.Sequential([
        # If the input shape hasen't been flattened yet
        layers.Flatten(input_shape=input_shape),

        # HIDDEN LAYER 1
        # High neuron count needed to capture complex patterns from raw pixels
        layers.Dense(1000, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)),

        # HIDDEN LAYER 2
        layers.Dense(700, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)),


        # HIDDEN LAYER 3
        layers.Dense(500, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001)),

     
        # OUTPUT LAYER
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_best_cnn_model(input_shape=(128, 128, 3)):
    """
    Builds the CNN model using the fixed 'Best Hyperparameters' found during tuning.
    """
    num_classes = 9 

    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # --- Convolutional Blocks ---
    
    # Block 1: Fixed to Kernel Size 3, Filters 64, Activation ReLU
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    ))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2: Fixed to Filters 64
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    ))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3: Fixed to Filters 128
    model.add(layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    ))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 4: 'Use Block 4' was True, Filters 192
    model.add(layers.Conv2D(
        filters=192,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    ))
    model.add(layers.MaxPooling2D((2, 2)))

    # --- Classification Head ---
    model.add(layers.Flatten())
    
    # Dense Layer: Fixed to 256 units
    model.add(layers.Dense(
        units=256,
        activation='relu'
    ))
    


    # Dropout: Fixed to 0.5
    model.add(layers.Dropout(rate=0.5))
    
    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    # --- Compilation ---
    # Optimizer: Adam, Learning Rate: 0.0005
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model