#!/usr/bin/env python3
"""
Multimodal prediction model training script
Combines CNN features from tuyere images with GRU temporal features from blast furnace parameters

python src/train_multimodal.py --dataset TuyereData/DataSetTrain_hourly
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.experimental import CosineDecay

from pyimagesearch import datasets, models
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, median_absolute_error

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multimodal prediction model training")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                       help="Input dataset path")
    parser.add_argument("--epochs", type=int, default=200,
                       help="Number of training epochs (default: 210)")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size (default: 2)")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                       help="Learning rate (default: 0.0001)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Training set ratio (default: 0.8)")
    return parser.parse_args()


def load_and_prepare_data(dataset_path, train_ratio=0.8):
    """Load and preprocess data"""
    print("[INFO] Loading blast furnace parameter data...")
    input_path = os.path.sep.join([dataset_path, "BF_paramters_687_half.csv"])
    df, features, labels = datasets.load_BFPsequence_data(input_path)
    
    print(f'Feature dimensions: {features.shape}')
    print(f'Label dimensions: {labels.shape}')
    
    print("[INFO] Loading tuyere image data...")
    images = datasets.load_Tuyere_images_hourly(df, dataset_path)
    print("[INFO] Data preprocessing...")
    
    # Split dataset by temporal order (8:1:1 ratio)
    train_split = int(len(features) * train_ratio)
    val_split = int(len(features) * (train_ratio + 0.1))
    
    train_attr_features = features[:train_split]
    val_attr_features = features[train_split:val_split]
    test_attr_features = features[val_split:]
    train_labels = labels[:train_split]
    val_labels = labels[train_split:val_split]
    test_labels = labels[val_split:]
    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]
    
    print(f'Training image dimensions: {train_images.shape}')
    print(f'Validation image dimensions: {val_images.shape}')
    print(f'Test image dimensions: {test_images.shape}')
    
    # Data standardization with persistent scaler
    scaler_path = "scalers/multimodal_scaler.pkl"
    train_attr_x, test_attr_x, train_y, test_y, _ = datasets.process_3Ddata_Scaler(
        train_attr_features, test_attr_features, train_labels, test_labels,
        scaler_path=scaler_path, mode='fit')
    val_attr_x, _, val_y, _, _ = datasets.process_3Ddata_Scaler(
        train_attr_features, val_attr_features, train_labels, val_labels,
        scaler_path=scaler_path, mode='transform')
    
    print(f'Processed training feature dimensions: {train_attr_x.shape}')
    print(f'Processed validation feature dimensions: {val_attr_x.shape}')
    print(f'Processed test feature dimensions: {test_attr_x.shape}')
    print(f'Processed training label dimensions: {train_y.shape}')
    print(f'Processed validation label dimensions: {val_y.shape}')
    print(f'Processed test label dimensions: {test_y.shape}')
    
    return (train_attr_x, val_attr_x, test_attr_x, train_y, val_y, test_y, 
            train_images, val_images, test_images, 
            train_attr_features, val_attr_features, test_attr_features, 
            train_labels, val_labels, test_labels)


def create_multimodal_model(attr_input_shape, image_input_shape=(320, 320, 3)):
    """Create multimodal fusion model"""
    # Create temporal feature extraction model (GRU) - matches baseline architecture
    gru_model = models.create_GRU_Simple_model(attr_input_shape)
    
    # Create image feature extraction model (CNN) - includes built-in normalization
    cnn_model = models.create_TSCNN_Model(image_input_shape)
    
    # Fuse features from both modalities
    combined_input = concatenate([gru_model.output, cnn_model.output])
    print(f'Fused feature dimensions: {combined_input.shape}')
    
    # Final fully connected layers
    x = Dense(4, activation="relu")(combined_input)
    x = Dense(1, activation="linear")(x)
    
    # Build complete model
    model = Model(inputs=[gru_model.input, cnn_model.input], outputs=x)
    
    return model


def setup_callbacks(checkpoint_dir="logs/Training_TSCNNGRU/0.5"):
    """Setup training callbacks"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, 'val_loss_{val_loss:.4f}_epoch_{epoch:02d}.h5')
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        monitor="val_loss",
        mode='min',
        save_best_only=True
    )
    
    return [checkpoint]


def calculate_hit_rate(actual_values, predicted_values, threshold=10):
    """Calculate hit rate"""
    hits = sum([1 if abs(actual - pred) < threshold else 0 
               for actual, pred in zip(actual_values, predicted_values)])
    total_predictions = len(predicted_values)
    hit_rate = (hits / total_predictions) * 100 if total_predictions > 0 else 0
    return hit_rate


def evaluate_model(model, test_attr_x, test_images, test_y, 
                  train_attr_features, test_attr_features, 
                  train_labels, test_labels):
    """Evaluate model performance"""
    print("[INFO] Making model predictions...")
    
    # Test set predictions
    preds = model.predict([test_attr_x, test_images])
    pred = preds.flatten()
    
    # Inverse transform predictions to original scale
    _, _, _, _, pred = datasets.process_3Ddata_Scaler(
        train_attr_features, test_attr_features, train_labels, test_labels, pred)
    _, _, _, _, test_y_original = datasets.process_3Ddata_Scaler(
        train_attr_features, test_attr_features, train_labels, test_labels, test_y)
    
    # Calculate evaluation metrics
    r2 = r2_score(test_y_original, pred)
    mse = mean_squared_error(test_y_original, pred)
    mae = mean_absolute_error(test_y_original, pred)
    median_ae = median_absolute_error(test_y_original, pred)
    explained_var = explained_variance_score(test_y_original, pred)
    hit_rate = calculate_hit_rate(test_y_original, pred)
    
    print("\n=== Model Evaluation Results ===")
    print(f'R² Score: {r2:.4f}')
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'Median Absolute Error: {median_ae:.4f}')
    print(f'Explained Variance Score: {explained_var:.4f}')
    print(f'Hit Rate (±10°C): {hit_rate:.2f}%')
    
    return pred, test_y_original


def plot_training_history(history):
    """Plot training history"""
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_predictions(pred, test_y, title_suffix="Test"):
    """Plot prediction results"""
    df_time = range(len(test_y))
    plt.figure(figsize=(10, 6))
    plt.title(f'HMT Predicted vs Actual Values ({title_suffix})', fontsize=14)
    plt.plot(df_time, test_y, alpha=0.7, color='red', label='Actual Values')
    plt.plot(df_time, pred, alpha=0.8, color='green', label=f'Predicted Values ({title_suffix})')
    plt.ylabel('HMT Temperature/°C')
    plt.xlabel('Tapping Number')
    plt.legend()
    plt.show()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Load and preprocess data
    (train_attr_x, val_attr_x, test_attr_x, train_y, val_y, test_y, 
     train_images, val_images, test_images, 
     train_attr_features, val_attr_features, test_attr_features, 
     train_labels, val_labels, test_labels) = load_and_prepare_data(
        args.dataset, args.train_ratio)
    
    # Create multimodal model
    attr_input_shape = (train_attr_x.shape[1], train_attr_x.shape[2])
    model = create_multimodal_model(attr_input_shape)
    
    # Compile model
    optimizer = Adam(learning_rate=args.learning_rate)
    model.compile(loss="MSE", optimizer=optimizer)
    
    # Display model architecture
    model.summary()
    
    # Setup callbacks
    callbacks = setup_callbacks()
    
    # Train model
    print("[INFO] Starting model training...")
    start_time = time.time()
    
    history = model.fit(
        [train_attr_x, train_images], train_y,
        validation_data=([val_attr_x, val_images], val_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in: {training_time:.2f} seconds")
    
    # Save complete model
    os.makedirs('Save_model', exist_ok=True)
    model.save('Save_model/multimodal_cnn_gru_model.h5')
    print("Model saved to: Save_model/multimodal_cnn_gru_model.h5")
    
    # Evaluate model
    pred, test_y_original = evaluate_model(
        model, test_attr_x, test_images, test_y,
        train_attr_features, test_attr_features, train_labels, test_labels)
    
    # Plot results
    plot_training_history(history)
    plot_predictions(pred, test_y_original)
    
    # Save prediction results
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame({
        "Difference": pred - test_y_original,
        "Predicted": pred,
        "Actual": test_y_original
    })
    results_path = "results/multimodal_predictions.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Prediction results saved to: {results_path}")


if __name__ == "__main__":
    main()
