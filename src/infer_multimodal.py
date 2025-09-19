#!/usr/bin/env python3
"""
Multimodal prediction model inference script
Load trained multimodal model for blast furnace thermal state prediction

Usage:
python src/infer_multimodal.py --dataset TuyereData/DataSetTrain_halfHourly --model-path logs/Training_TSCNNGRU/0.5/val_loss_0.0080_epoch_192.h5
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import load_model
from pyimagesearch import datasets, models
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score, median_absolute_error


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multimodal prediction model inference")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                       help="Input dataset path")
    parser.add_argument("-m", "--model-path", type=str, 
                       default="Save_model/multimodal_cnn_gru_model.h5",
                       help="Trained model path")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Training set ratio for data splitting (default: 0.8)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Results output directory (default: results)")
    parser.add_argument("--save-predictions", action="store_true",
                       help="Whether to save prediction results to CSV file")
    parser.add_argument("--scaler-path", type=str, default="scalers/multimodal_scaler.pkl",
                       help="Path to saved scaler file (default: scalers/multimodal_scaler.pkl)")
    return parser.parse_args()


def load_and_prepare_data(dataset_path, train_ratio=0.8, scaler_path=None):
    """Load and preprocess data with 8:1:1 split"""
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
    
    # Data standardization using saved scaler
    train_attr_x, test_attr_x, train_y, test_y, _ = datasets.process_3Ddata_Scaler(
        train_attr_features, test_attr_features, train_labels, test_labels,
        scaler_path=scaler_path, mode='transform')
    val_attr_x, _, val_y, _, _ = datasets.process_3Ddata_Scaler(
        train_attr_features, val_attr_features, train_labels, val_labels,
        scaler_path=scaler_path, mode='transform')
    
    print(f'Processed training feature dimensions: {train_attr_x.shape}')
    print(f'Processed validation feature dimensions: {val_attr_x.shape}')
    print(f'Processed test feature dimensions: {test_attr_x.shape}')
    
    return (train_attr_x, val_attr_x, test_attr_x, train_y, val_y, test_y, 
            train_images, val_images, test_images, 
            train_attr_features, val_attr_features, test_attr_features, 
            train_labels, val_labels, test_labels)


def load_model_safe(model_path, custom_objects=None):
    """Safely load model with custom objects"""
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"Successfully loaded model: {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Cannot load model {model_path}: {e}")


def calculate_hit_rate(actual_values, predicted_values, threshold=10):
    """Calculate hit rate"""
    hits = sum([1 if abs(actual - pred) < threshold else 0 
               for actual, pred in zip(actual_values, predicted_values)])
    total_predictions = len(predicted_values)
    hit_rate = (hits / total_predictions) * 100 if total_predictions > 0 else 0
    return hit_rate


def evaluate_predictions(y_true, y_pred, dataset_name="Dataset"):
    """Evaluate prediction results"""
    print(f"\n=== {dataset_name} Evaluation Results ===")
    print(f'R² Score: {r2_score(y_true, y_pred):.4f}')
    print(f'Mean Squared Error (MSE): {mean_squared_error(y_true, y_pred):.4f}')
    print(f'Mean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred):.4f}')
    print(f'Median Absolute Error: {median_absolute_error(y_true, y_pred):.4f}')
    print(f'Explained Variance Score: {explained_variance_score(y_true, y_pred):.4f}')
    print(f'Hit Rate (±10°C): {calculate_hit_rate(y_true, y_pred):.2f}%')


def plot_predictions(pred, true_values, title="Dataset", save_path=None):
    """Plot prediction results"""
    df_time = range(len(true_values))
    plt.figure(figsize=(10, 6))
    plt.title(f'HMT Predicted vs Actual Values ({title})', fontsize=14)
    plt.plot(df_time, true_values, alpha=0.7, color='red', label='Actual Values')
    plt.plot(df_time, pred, alpha=0.8, color='green', label='Predicted Values')
    plt.ylabel('HMT Temperature/°C')
    plt.xlabel('Tapping Number')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to: {save_path}")
    
    plt.show()


def plot_scatter(pred, true_values, title="Predicted vs Actual Values", xlim=None, ylim=None, save_path=None):
    """Plot scatter plot"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw scatter points
    ax.scatter(true_values, pred, edgecolors='blue', alpha=0.6)
    
    # Add ideal prediction line (y=x)
    if xlim is None:
        xlim = (min(true_values.min(), pred.min()), max(true_values.max(), pred.max()))
    if ylim is None:
        ylim = xlim
    
    ax.plot(xlim, ylim, linestyle='--', linewidth=1.5, color='red', alpha=0.8)
    
    ax.set_xlabel('Actual Values/°C')
    ax.set_ylabel('Predicted Values/°C')
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to: {save_path}")
    
    plt.show()


def save_predictions(pred, true_values, output_path):
    """Save prediction results to CSV file"""
    results_df = pd.DataFrame({
        "Difference": pred - true_values,
        "Predicted": pred,
        "Actual": true_values
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Prediction results saved to: {output_path}")


def main():
    """Main function"""
    args = parse_arguments()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file does not exist - {args.model_path}")
        return
    
    # Load and preprocess data
    (train_attr_x, val_attr_x, test_attr_x, train_y, val_y, test_y, 
     train_images, val_images, test_images, 
     train_attr_features, val_attr_features, test_attr_features, 
     train_labels, val_labels, test_labels) = load_and_prepare_data(
        args.dataset, args.train_ratio, args.scaler_path)
    
    # Define custom objects for model loading
    custom_objects = {"AttentionLayer": models.AttentionLayer}
    
    # Load trained model with custom objects
    model = load_model_safe(args.model_path, custom_objects=custom_objects)
    model.summary()
    
    print("[INFO] Performing model inference...")
    
    # Test set predictions
    test_preds = model.predict([test_attr_x, test_images])
    test_pred = test_preds.flatten()
    
    # Validation set predictions
    val_preds = model.predict([val_attr_x, val_images])
    val_pred = val_preds.flatten()
    
    # Training set predictions
    train_preds = model.predict([train_attr_x, train_images])
    train_pred = train_preds.flatten()
    
    # Inverse transform predictions to original scale
    _, _, _, _, test_pred_original = datasets.process_3Ddata_Scaler(
        train_attr_features, test_attr_features, train_labels, test_labels, test_pred)
    _, _, _, _, test_y_original = datasets.process_3Ddata_Scaler(
        train_attr_features, test_attr_features, train_labels, test_labels, test_y)
    
    _, _, _, _, val_pred_original = datasets.process_3Ddata_Scaler(
        train_attr_features, val_attr_features, train_labels, val_labels, val_pred)
    _, _, _, _, val_y_original = datasets.process_3Ddata_Scaler(
        train_attr_features, val_attr_features, train_labels, val_labels, val_y)
    
    _, _, _, _, train_pred_original = datasets.process_3Ddata_Scaler(
        train_attr_features, test_attr_features, train_labels, test_labels, train_pred)
    _, _, _, _, train_y_original = datasets.process_3Ddata_Scaler(
        train_attr_features, test_attr_features, train_labels, test_labels, train_y)
    
    # Evaluate prediction results
    evaluate_predictions(train_y_original, train_pred_original, "Training Set")
    evaluate_predictions(val_y_original, val_pred_original, "Validation Set")
    evaluate_predictions(test_y_original, test_pred_original, "Test Set")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot prediction results
    plot_predictions(test_pred_original, test_y_original, "Test Set",
                    os.path.join(args.output_dir, "multimodal_test_predictions.png"))
    
    plot_predictions(val_pred_original, val_y_original, "Validation Set",
                    os.path.join(args.output_dir, "multimodal_val_predictions.png"))
    
    # Plot scatter plot
    plot_scatter(test_pred_original, test_y_original, 
                xlim=(1430, 1500), ylim=(1430, 1500),
                save_path=os.path.join(args.output_dir, "multimodal_scatter_plot.png"))
    
    # Plot combined training, validation and test set scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(train_y_original, train_pred_original, color='blue', label='Training Data', alpha=0.6)
    plt.scatter(val_y_original, val_pred_original, color='orange', label='Validation Data', alpha=0.6)
    plt.scatter(test_y_original, test_pred_original, color='red', label='Test Data', alpha=0.6)
    
    # Add ideal prediction line
    min_val = min(min(train_y_original), min(val_y_original), min(test_y_original))
    max_val = max(max(train_y_original), max(val_y_original), max(test_y_original))
    plt.plot([min_val, max_val], [min_val, max_val], color='grey', linestyle='--')
    
    plt.title('Actual vs Predicted Values (Multimodal Model)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    combined_scatter_path = os.path.join(args.output_dir, "multimodal_combined_scatter.png")
    plt.savefig(combined_scatter_path, dpi=300, bbox_inches='tight')
    print(f"Combined scatter plot saved to: {combined_scatter_path}")
    plt.show()
    
    # Save prediction results
    if args.save_predictions:
        save_predictions(test_pred_original, test_y_original,
                        os.path.join(args.output_dir, "multimodal_test_predictions.csv"))
        save_predictions(val_pred_original, val_y_original,
                        os.path.join(args.output_dir, "multimodal_val_predictions.csv"))


if __name__ == "__main__":
    main()