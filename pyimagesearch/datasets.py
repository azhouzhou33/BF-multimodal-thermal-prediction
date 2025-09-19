#!/usr/bin/env python3
"""
Dataset loading and preprocessing utilities
Handles blast furnace parameter sequence data and tuyere image loading with proper scaling
"""

import os
import pickle
import numpy as np
import pandas as pd
import cv2
import glob
import re
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import img_to_array


def load_BFPsequence_data(inputPath, feature_steps=6):
    """
    Load blast furnace parameter sequence data from CSV
    
    Args:
        inputPath: Path to CSV file containing time series data
        feature_steps: Number of time steps for sequence features (default: 6 to match paper)
    
    Returns:
        df: DataFrame with metadata
        features: Time series features array (N, feature_steps, num_features)
        labels: Target labels array (N,)
    """
    # Read CSV file to DataFrame
    df = pd.read_csv(inputPath)
    data = df.values  # Convert DataFrame to numpy array

    # Define feature and label lists
    features = []
    labels = []
    label_step = 1

    # Extract features and labels by traversing data
    for i in range(len(data) - feature_steps - label_step + 1):
        # Extract consecutive feature_steps time steps starting from current time step as features
        feature = data[i:(i + feature_steps), :-1]
        # Extract data label_step time steps later as label
        label = data[i + feature_steps + label_step - 1, -1]

        features.append(feature)
        labels.append(label)

    # Convert lists to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    print(f"Loaded BFP sequence data: {features.shape} features, {labels.shape} labels")
    return df, features, labels


def load_Tuyere_images_hourly(df, inputPath, num_images_per_hour=6):
    """
    Load tuyere images grouped by hour (6 frames/hour as per paper)
    
    Args:
        df: DataFrame with image metadata
        inputPath: Path to dataset directory containing images
        num_images_per_hour: Number of images per hour (default: 6)
    
    Returns:
        images: Array of image sequences (N, num_images_per_hour, H, W, C) with /255 normalization
    """
    # Initialize array to store image data
    images_data = []
    
    # Calculate total images needed based on actual df length, not hardcoded value
    num_hours = len(df)
    total_images_needed = num_hours * num_images_per_hour
    
    print(f'DataFrame length: {len(df)}')
    print(f'Total images needed: {total_images_needed}')
    
    # Load images
    for i in range(total_images_needed):
        # Find image files matching the pattern
        image_pattern = os.path.sep.join([inputPath, f"{i}_*.png"])
        image_files = glob.glob(image_pattern)
        
        if not image_files:
            # If no image found, create a placeholder
            print(f"Warning: No image found for index {i}, using placeholder")
            placeholder = np.zeros((320, 320, 3), dtype=np.float32)
            images_data.append(placeholder)
            continue
        
        # Sort files to ensure consistent order
        image_files.sort(key=lambda x: int(re.search(r'(\d+)_', x).group(1)))
        
        # Load the first matching image
        image_file = image_files[0]
        
        # Ensure file exists
        if not os.path.isfile(image_file):
            print(f"Warning: File does not exist: {image_file}, using placeholder")
            placeholder = np.zeros((320, 320, 3), dtype=np.float32)
            images_data.append(placeholder)
            continue
            
        # Read and preprocess image
        image = cv2.imread(image_file)
        if image is not None:
            # Resize to standard size and convert to array
            image = cv2.resize(image, (320, 320))
            img_array = img_to_array(image)
            # Normalize pixel values to [0, 1] as per paper requirements
            img_array = img_array / 255.0
            images_data.append(img_array)
        else:
            print(f"Warning: Failed to read image: {image_file}, using placeholder")
            placeholder = np.zeros((320, 320, 3), dtype=np.float32)
            images_data.append(placeholder)

        # Stop if we've collected enough images
        if len(images_data) == total_images_needed:
            break

    # Group image data by hour (6 images per hour)
    hourly_batches = []
    for i in range(0, len(images_data), num_images_per_hour):
        batch = images_data[i:i + num_images_per_hour]
        # Pad batch if insufficient images
        while len(batch) < num_images_per_hour:
            batch.append(np.zeros((320, 320, 3), dtype=np.float32))
        hourly_batches.append(batch)

    # Convert to numpy array: (N, 6, H, W, C) - sequence dimension first as per paper
    input_data = np.array(hourly_batches)
    
    print(f"Loaded tuyere images shape: {input_data.shape}")
    return input_data


def process_3Ddata_Scaler(train_X, test_X, labels_train, labels_test, pred_value=None, scaler_path=None, mode='fit'):
    """
    Process 3D time series data with MinMax scaling and persistent scaler support
    
    Args:
        train_X: Training features (N, timesteps, features)
        test_X: Test features (N, timesteps, features)
        labels_train: Training labels
        labels_test: Test labels
        pred_value: Predictions to inverse transform (optional)
        scaler_path: Path to save/load scaler (optional)
        mode: 'fit' for training, 'transform' for inference
    
    Returns:
        Tuple of (train_X_scaled, test_X_scaled, labels_train_scaled, labels_test_scaled, pred_inverse)
    """
    
    if mode == 'fit' or scaler_path is None:
        # Initialize scalers
        feature_scaler = MinMaxScaler()
        labels_scaler = MinMaxScaler()
        
        # Reshape 3D data to 2D for scaling, then reshape back
        n_samples, n_windows, n_features = train_X.shape
        features_reshaped = train_X.reshape(n_samples * n_windows, n_features)
        
        # Fit scaler on training data and transform
        features_reshaped = feature_scaler.fit_transform(features_reshaped)
        train_X_scaled = features_reshaped.reshape(n_samples, n_windows, n_features)
        
        # Scale labels
        labels_train_scaled = labels_scaler.fit_transform(labels_train.reshape(-1, 1)).ravel()
        labels_test_scaled = labels_scaler.transform(labels_test.reshape(-1, 1)).ravel()
        
        # Scale test features (transform only, no fit)
        n_samples_test, n_windows, n_features = test_X.shape
        X_test_reshaped = test_X.reshape(n_samples_test * n_windows, n_features)
        X_test_reshaped = feature_scaler.transform(X_test_reshaped)
        test_X_scaled = X_test_reshaped.reshape(n_samples_test, n_windows, n_features)
        
        # Save scalers if path provided
        if scaler_path:
            scaler_data = {
                'feature_scaler': feature_scaler,
                'labels_scaler': labels_scaler,
                'train_shape': train_X.shape,
                'test_shape': test_X.shape
            }
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler_data, f)
            print(f"Scalers saved to {scaler_path}")
            
    else:
        # Load existing scalers
        try:
            with open(scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)
            feature_scaler = scaler_data['feature_scaler']
            labels_scaler = scaler_data['labels_scaler']
            print(f"Scalers loaded from {scaler_path}")
            
            # Transform data using loaded scalers
            n_samples, n_windows, n_features = train_X.shape
            features_reshaped = train_X.reshape(n_samples * n_windows, n_features)
            features_reshaped = feature_scaler.transform(features_reshaped)
            train_X_scaled = features_reshaped.reshape(n_samples, n_windows, n_features)
            
            n_samples_test, n_windows, n_features = test_X.shape
            X_test_reshaped = test_X.reshape(n_samples_test * n_windows, n_features)
            X_test_reshaped = feature_scaler.transform(X_test_reshaped)
            test_X_scaled = X_test_reshaped.reshape(n_samples_test, n_windows, n_features)
            
            labels_train_scaled = labels_scaler.transform(labels_train.reshape(-1, 1)).ravel()
            labels_test_scaled = labels_scaler.transform(labels_test.reshape(-1, 1)).ravel()
            
        except Exception as e:
            print(f"Error loading scalers from {scaler_path}: {e}")
            print("Falling back to fitting new scalers...")
            return process_3Ddata_Scaler(
                train_X, test_X, labels_train, labels_test, pred_value, None, 'fit'
            )

    # Handle prediction inverse transform
    pred_inverse = None
    if pred_value is not None:
        pred_value_np = np.array(pred_value)
        if mode == 'fit' or scaler_path is None:
            pred_inverse = labels_scaler.inverse_transform(pred_value_np.reshape(-1, 1)).ravel()
        else:
            pred_inverse = labels_scaler.inverse_transform(pred_value_np.reshape(-1, 1)).ravel()

    return train_X_scaled, test_X_scaled, labels_train_scaled, labels_test_scaled, pred_inverse


def process_2Ddata_Scaler(train_X, test_X, labels_train, labels_test, pred_value=None, scaler_path=None, mode='fit'):
    """
    Process 2D data with MinMax scaling and persistent scaler support
    
    Args:
        train_X: Training features (N, features)
        test_X: Test features (N, features)
        labels_train: Training labels
        labels_test: Test labels
        pred_value: Predictions to inverse transform (optional)
        scaler_path: Path to save/load scaler (optional)
        mode: 'fit' for training, 'transform' for inference
    
    Returns:
        Tuple of (train_X_scaled, test_X_scaled, labels_train_scaled, labels_test_scaled, pred_inverse)
    """
    
    if mode == 'fit' or scaler_path is None:
        # Initialize scalers
        feature_scaler = MinMaxScaler()
        labels_scaler = MinMaxScaler()
        
        # Fit and transform training data
        train_X_scaled = feature_scaler.fit_transform(train_X)
        
        # Scale labels
        labels_train_scaled = labels_scaler.fit_transform(labels_train.reshape(-1, 1)).ravel()
        labels_test_scaled = labels_scaler.transform(labels_test.reshape(-1, 1)).ravel()
        
        # Scale test features (transform only)
        test_X_scaled = feature_scaler.transform(test_X)
        
        # Save scalers if path provided
        if scaler_path:
            scaler_data = {
                'feature_scaler': feature_scaler,
                'labels_scaler': labels_scaler
            }
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler_data, f)
            print(f"Scalers saved to {scaler_path}")
            
    else:
        # Load existing scalers
        try:
            with open(scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)
            feature_scaler = scaler_data['feature_scaler']
            labels_scaler = scaler_data['labels_scaler']
            print(f"Scalers loaded from {scaler_path}")
            
            # Transform data using loaded scalers
            train_X_scaled = feature_scaler.transform(train_X)
            test_X_scaled = feature_scaler.transform(test_X)
            labels_train_scaled = labels_scaler.transform(labels_train.reshape(-1, 1)).ravel()
            labels_test_scaled = labels_scaler.transform(labels_test.reshape(-1, 1)).ravel()
            
        except Exception as e:
            print(f"Error loading scalers from {scaler_path}: {e}")
            print("Falling back to fitting new scalers...")
            return process_2Ddata_Scaler(
                train_X, test_X, labels_train, labels_test, pred_value, None, 'fit'
            )

    # Handle prediction inverse transform
    pred_inverse = None
    if pred_value is not None:
        pred_value_np = np.array(pred_value)
        if mode == 'fit' or scaler_path is None:
            pred_inverse = labels_scaler.inverse_transform(pred_value_np.reshape(-1, 1)).ravel()
        else:
            pred_inverse = labels_scaler.inverse_transform(pred_value_np.reshape(-1, 1)).ravel()

    return train_X_scaled, test_X_scaled, labels_train_scaled, labels_test_scaled, pred_inverse