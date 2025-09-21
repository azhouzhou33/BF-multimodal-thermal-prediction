# Multimodal Blast Furnace Thermal State Prediction Model

## Project Overview

This project implements a multimodal prediction model that combines CNN features from tuyere images (6 frames/hour) with GRU temporal features from blast furnace operational parameters. The goal is to achieve accurate prediction of blast furnace thermal state indicators and demonstrate the advantages of image-parameter joint modeling over single data source approaches.

## Project Structure

```
Code/
├── src/                           # Source code directory
│   ├── train_multimodal.py        # Multimodal model training script
│   ├── infer_multimodal.py        # Multimodal model inference script
│   ├── train_baseline_gru.py      # Baseline GRU model training script
│   └── infer_baseline_gru.py      # Baseline GRU model inference script
├── pyimagesearch/                 # Core modules
│   ├── models.py                  # Model architectures (TSCNN, GRU, LSTM)
│   └── datasets.py                # Data loading and preprocessing utilities
├── scalers/                       # Saved scaler objects for consistent preprocessing
├── logs/                          # Training logs and model checkpoints
├── Save_model/                    # Saved complete models
├── results/                       # Prediction results and visualizations
└── README.md                      # Project documentation
```


## Model Architecture

### Multimodal Pipeline
```
Time Series Input (batch, 6, features) → GRU_Simple → 4D temporal features ┐
                                                                          ├→ Concatenate → Dense → Prediction
Image Sequence Input (batch, 6, 320, 320, 3) → TSCNN → 4D image features ┘
```

### Baseline Pipeline
```
Time Series Input (batch, 6, features) → GRU_Complete → Direct Prediction
```

### TSCNN Details
- **Input**: `(batch, 6, 320, 320, 3)` - 6 frames per hour
- **Processing**: Pairwise depth concatenation (frames 0+1, 2+3, 4+5) → individual convolution
- **Output**: 4D feature vector for fusion with GRU features

### GRU Details
- **Architecture**: 256→128→64→32 GRU units with attention mechanism
- **Output**: 4D temporal feature vector (multimodal) or direct prediction (baseline)

## Usage Instructions

### Environment Requirements
```bash
pip install tensorflow keras scikit-learn matplotlib pandas numpy opencv-python
```

### Training Models

#### Multimodal Model Training
```bash
python src/train_multimodal.py --dataset TuyereData/DataSetTrain_hourly --epochs 200 --batch-size 64 --learning-rate 0.0001
```

#### Baseline Model Training
```bash
python src/train_baseline_gru.py --dataset TuyereData/DataSetTrain_hour --epochs 200 --model-type gru --learning-rate 0.0001
```

### Model Inference

#### Multimodal Model Inference
```bash
python src/infer_multimodal.py --dataset TuyereData/DataSetTrain_Hourly --model-path Save_model/multimodal_cnn_gru_model.h5 --save-predictions
```

#### Baseline Model Inference
```bash
python src/infer_baseline_gru.py --dataset TuyereData/DataSetTrain_Hourly --model-path Save_model/baseline_gru_model.h5 --save-predictions
```

## Parameter Reference

### Training Parameters
- `--dataset`: Path to dataset directory containing CSV and image files
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--learning-rate`: Learning rate (default: 0.0001)
- `--train-ratio`: Training set ratio (default: 0.8 for 8:1:1 split)
- `--model-type`: Model type for baseline (choices: gru, lstm)

### Inference Parameters
- `--model-path`: Path to trained model file
- `--scaler-path`: Path to saved scaler file
- `--output-dir`: Directory for saving results
- `--save-predictions`: Flag to save prediction results to CSV

## Data Requirements

### Expected Data Structure
```
TuyereData/DataSetTrain_hourly/
├── BF_paramters_687_half.csv     # Time series data
├── 0_Tuyere.png                  # Image files
├── 1_Tuyere.png
├── ...
└── N_Tuyere.png
```

### CSV Format
- **Features**: Time series operational parameters (multiple columns)
- **Target**: Last column should contain the target variable (HMT temperature)
- **Temporal Order**: Data should be ordered chronologically

### Image Format
- **Resolution**: Images will be resized to 320×320×3
- **Naming**: Sequential naming pattern: `{index}_Tuyere.png`
- **Grouping**: 6 consecutive images per hour for temporal modeling

## Evaluation Metrics

The model performance is evaluated using:

- **R² Score**: Model goodness of fit
- **Mean Squared Error (MSE)**: Average squared prediction error
- **Mean Absolute Error (MAE)**: Average absolute prediction deviation
- **Median Absolute Error**: Robust measure of central tendency error
- **Explained Variance Score**: Proportion of variance explained by the model
- **Hit Rate**: Percentage of predictions within ±10°C of actual values


## Training Output
- **Model Checkpoints**: `logs/Training_*/` with best validation loss models
- **Complete Models**: `Save_model/` with full trained models
- **Training History**: Loss curves and performance metrics



## Performance Optimization

1. **GPU Usage**: Modify `CUDA_VISIBLE_DEVICES` in scripts to use specific GPUs
2. **Batch Size**: Increase batch size if memory allows for faster training
3. **Data Loading**: Images are loaded and processed efficiently with built-in normalization

## Citation

If you use this code in your research, please cite the corresponding paper that describes the multimodal blast furnace thermal state prediction methodology.

## License

This project is provided for academic and research purposes. Please refer to the specific license terms if applicable.
