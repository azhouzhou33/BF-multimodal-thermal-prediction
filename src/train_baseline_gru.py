#!/usr/bin/env python3
"""
Baseline GRU model training script
Uses only blast furnace operational parameter time series features for modeling,
serving as a baseline comparison for the multimodal model

Usage:
python src/train_baseline_gru.py --dataset TuyereData/DataSetTrain_hour
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.experimental import CosineDecay

from pyimagesearch import datasets, models
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, median_absolute_error

# Set to use CPU training (modify to corresponding GPU number if GPU is needed)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Baseline GRU model training")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                       help="Input dataset path")
    parser.add_argument("--epochs", type=int, default=200,
                       help="Number of training epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size (default: 2)")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                       help="Learning rate (default: 0.0001)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Training set ratio (default: 0.8)")
    parser.add_argument("--model-type", type=str, choices=['gru', 'lstm'], default='gru',
                       help="Model type (default: gru)")
    return parser.parse_args()


def load_and_prepare_data(dataset_path, train_ratio=0.8):
    """Load and preprocess data"""
    print("[INFO] 加载高炉参数数据...")
    input_path = os.path.sep.join([dataset_path, "BF_paramters_687_half.csv"])
    df, features, labels = datasets.load_BFPsequence_data(input_path)
    
    print(f'特征维度: {features.shape}')
    print(f'标签维度: {labels.shape}')
    
    # Split dataset by temporal order (8:1:1 ratio)
    train_split = int(len(features) * train_ratio)
    val_split = int(len(features) * (train_ratio + 0.1))
    
    train_attr_features = features[:train_split]
    val_attr_features = features[train_split:val_split]
    test_attr_features = features[val_split:]
    train_labels = labels[:train_split]
    val_labels = labels[train_split:val_split]
    test_labels = labels[val_split:]
    
    # Data standardization with persistent scaler
    scaler_path = "scalers/baseline_gru_scaler.pkl"
    train_attr_x, test_attr_x, train_y, test_y, _ = datasets.process_3Ddata_Scaler(
        train_attr_features, test_attr_features, train_labels, test_labels,
        scaler_path=scaler_path, mode='fit')
    val_attr_x, _, val_y, _, _ = datasets.process_3Ddata_Scaler(
        train_attr_features, val_attr_features, train_labels, val_labels,
        scaler_path=scaler_path, mode='transform')
    
    print(f'处理后训练特征维度: {train_attr_x.shape}')
    print(f'处理后测试特征维度: {test_attr_x.shape}')
    print(f'处理后训练标签维度: {train_y.shape}')
    print(f'处理后测试标签维度: {test_y.shape}')
    
    return (train_attr_x, test_attr_x, train_y, test_y, 
            train_attr_features, test_attr_features, train_labels, test_labels)


def create_baseline_model(input_shape, model_type='gru'):
    """Create baseline temporal model"""
    if model_type.lower() == 'gru':
        model = models.create_GRU_model(input_shape)
        print("Creating GRU baseline model")
    elif model_type.lower() == 'lstm':
        model = models.create_lstm_model(input_shape)
        print("Creating LSTM baseline model")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model


def setup_callbacks(model_type='gru', checkpoint_dir=None):
    """Setup training callbacks"""
    if checkpoint_dir is None:
        checkpoint_dir = f"logs/training_{model_type.upper()}/1.5"
    
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


def evaluate_model(model, train_attr_x, test_attr_x, train_y, test_y,
                  train_attr_features, test_attr_features, 
                  train_labels, test_labels):
    """Evaluate model performance"""
    print("[INFO] Making model predictions...")
    
    # Test set predictions
    test_preds = model.predict(test_attr_x)
    test_pred = test_preds.flatten()
    
    # Training set predictions
    train_preds = model.predict(train_attr_x)
    train_pred = train_preds.flatten()
    
    # Inverse transform predictions to original scale
    _, _, _, _, test_pred_original = datasets.process_3Ddata_Scaler(
        train_attr_features, test_attr_features, train_labels, test_labels, test_pred)
    _, _, _, _, test_y_original = datasets.process_3Ddata_Scaler(
        train_attr_features, test_attr_features, train_labels, test_labels, test_y)
    
    _, _, _, _, train_pred_original = datasets.process_3Ddata_Scaler(
        train_attr_features, test_attr_features, train_labels, test_labels, train_pred)
    _, _, _, _, train_y_original = datasets.process_3Ddata_Scaler(
        train_attr_features, test_attr_features, train_labels, test_labels, train_y)
    
    # Calculate evaluation metrics
    print("\n=== Test Set Evaluation Results ===")
    print(f'R² Score: {r2_score(test_y_original, test_pred_original):.4f}')
    print(f'Mean Squared Error (MSE): {mean_squared_error(test_y_original, test_pred_original):.4f}')
    print(f'Mean Absolute Error (MAE): {mean_absolute_error(test_y_original, test_pred_original):.4f}')
    print(f'Median Absolute Error: {median_absolute_error(test_y_original, test_pred_original):.4f}')
    print(f'Explained Variance Score: {explained_variance_score(test_y_original, test_pred_original):.4f}')
    print(f'Hit Rate (±10°C): {calculate_hit_rate(test_y_original, test_pred_original):.2f}%')
    
    print("\n=== Training Set Evaluation Results ===")
    print(f'R² Score: {r2_score(train_y_original, train_pred_original):.4f}')
    print(f'Mean Squared Error (MSE): {mean_squared_error(train_y_original, train_pred_original):.4f}')
    print(f'Mean Absolute Error (MAE): {mean_absolute_error(train_y_original, train_pred_original):.4f}')
    print(f'Hit Rate (±10°C): {calculate_hit_rate(train_y_original, train_pred_original):.2f}%')
    
    return train_pred_original, train_y_original, test_pred_original, test_y_original


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


def plot_predictions(pred, true_values, title_suffix=""):
    """Plot prediction results"""
    df_time = range(len(true_values))
    plt.figure(figsize=(10, 6))
    plt.title(f'HMT Predicted vs Actual Values ({title_suffix})', fontsize=14)
    plt.plot(df_time, true_values, alpha=0.7, color='red', label='Actual Values')
    plt.plot(df_time, pred, alpha=0.8, color='green', label=f'Predicted Values ({title_suffix})')
    plt.ylabel('HMT Temperature/°C')
    plt.xlabel('Tapping Number')
    plt.legend()
    plt.show()


def plot_scatter_comparison(train_y, train_pred, test_y, test_pred):
    """Plot training and test set comparison scatter plot"""
    plt.figure(figsize=(10, 8))
    plt.scatter(train_y, train_pred, color='blue', label='Training Data', alpha=0.6)
    plt.scatter(test_y, test_pred, color='red', label='Test Data', alpha=0.6)
    
    # Add ideal prediction line
    min_val = min(min(train_y), min(test_y))
    max_val = max(max(train_y), max(test_y))
    plt.plot([min_val, max_val], [min_val, max_val], color='grey', linestyle='--')
    
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Load and preprocess data
    (train_attr_x, val_attr_x, test_attr_x, train_y, val_y, test_y, 
     train_attr_features, val_attr_features, test_attr_features, 
     train_labels, val_labels, test_labels) = load_and_prepare_data(
        args.dataset, args.train_ratio)
    
    # Create baseline model
    input_shape = (train_attr_x.shape[1], train_attr_x.shape[2])
    model = create_baseline_model(input_shape, args.model_type)
    
    # Compile model
    optimizer = Adam(learning_rate=args.learning_rate)
    model.compile(loss="MSE", optimizer=optimizer)
    
    # Display model architecture
    model.summary()
    
    # Setup callbacks
    callbacks = setup_callbacks(args.model_type)
    
    # Train model
    print("[INFO] Starting baseline model training...")
    start_time = time.time()
    
    history = model.fit(
        train_attr_x, train_y,
        validation_data=(val_attr_x, val_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in: {training_time:.2f} seconds")
    
    # Save complete model
    os.makedirs('Save_model', exist_ok=True)
    model_save_path = f'Save_model/baseline_{args.model_type}_model.h5'
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Evaluate model
    train_pred, train_y_orig, test_pred, test_y_orig = evaluate_model(
        model, train_attr_x, test_attr_x, train_y, test_y,
        train_attr_features, test_attr_features, train_labels, test_labels)
    
    # Plot results
    plot_training_history(history)
    plot_predictions(train_pred, train_y_orig, "Training Set")
    plot_predictions(test_pred, test_y_orig, "Test Set")
    plot_scatter_comparison(train_y_orig, train_pred, test_y_orig, test_pred)
    
    # Save prediction results
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame({
        "Difference": test_pred - test_y_orig,
        "Predicted": test_pred,
        "Actual": test_y_orig
    })
    results_path = f"results/baseline_{args.model_type}_predictions.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Prediction results saved to: {results_path}")


if __name__ == "__main__":
    main()
