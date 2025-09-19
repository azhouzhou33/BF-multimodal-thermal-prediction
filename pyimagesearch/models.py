#!/usr/bin/env python3
"""
Model definitions for multimodal blast furnace prediction
Contains TSCNN, GRU, and LSTM model architectures aligned with paper specifications
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, GRU, LSTM, Conv2D, MaxPooling2D, 
    Flatten, Dropout, BatchNormalization, Activation,
    concatenate, Lambda, Layer
)


class AttentionLayer(Layer):
    """Custom attention layer for GRU models"""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = tf.keras.activations.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.activations.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def create_TSCNN_Model(input_shape=(320, 320, 3)):
    """
    Create Time Series CNN model for image feature extraction (6 frames/hour)
    Input shape: (batch, seq=6, H, W, C) - sequence dimension first as per paper
    Performs pairwise depth concatenation then convolution, flattening to generate 4D image features
    
    Args:
        input_shape: Single image shape (H, W, C), default (320, 320, 3)
    
    Returns:
        Keras model for image feature extraction outputting 4D features (not final scalar)
    """
    # Input expects sequence of 6 images: (batch, 6, H, W, C)
    inputs = Input(shape=(6, *input_shape))
    
    def conv_block(x, filters):
        """Convolutional block for processing individual images"""
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        return x
    
    # Process each image in the sequence through its own block with pairwise concatenation
    conv_blocks_outputs = []
    
    for i in range(6):  # 6 frames per hour as per paper
        # Extract i-th image from sequence
        image = Lambda(lambda x: x[:, i])(inputs)
        
        if i % 2 == 0:  # For even indices (0, 2, 4)
            # Process through conv block with increasing filters
            block_output = conv_block(image, 32 * (i // 2 + 1))
            conv_blocks_outputs.append(block_output)
        else:  # For odd indices (1, 3, 5) - pairwise depth concatenation
            # Process current image
            block_output = conv_block(image, 32 * (i // 2 + 1))
            # Concatenate with previous block output (depth concatenation)
            previous_block_output = conv_blocks_outputs[-1]
            merged_output = concatenate([previous_block_output, block_output], axis=-1)
            # Replace the last output with merged one
            conv_blocks_outputs[-1] = merged_output
    
    # Now we have 3 merged outputs from pairwise concatenation
    # Further processing of these merged outputs
    final_convs = []
    for merged_output in conv_blocks_outputs:
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(merged_output)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        final_convs.append(x)
    
    # Concatenate all final conv outputs
    concatenated = concatenate(final_convs, axis=-1)
    
    # Final layers to generate 4D image features (not final scalar regression)
    x = Dense(64, activation='relu')(concatenated)
    outputs = Dense(4, activation='relu')(x)  # 4D features for fusion
    
    model = Model(inputs=inputs, outputs=outputs, name='TSCNN_Model')
    return model


def create_GRU_Simple_model(input_shape, dropout_rate=0.3):
    """
    Create simple GRU model for multimodal fusion (feature extraction branch)
    Outputs 4D temporal features (not final scalar) for concatenation with TSCNN features
    
    Args:
        input_shape: Input shape (timesteps, features)
        dropout_rate: Dropout rate (default: 0.3)
    
    Returns:
        Keras model for time series feature extraction outputting 4D features
    """
    inputs = Input(shape=input_shape)
    
    # Multi-layer GRU with attention mechanism as per original implementation
    x = GRU(256, return_sequences=True)(inputs)
    x = Dropout(dropout_rate)(x)
    x = GRU(128, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x = GRU(64, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x = GRU(32, return_sequences=True)(x)
    
    # Add attention layer for temporal feature aggregation
    x = AttentionLayer()(x)
    
    # Output 4D temporal features (not final regression)
    outputs = Dense(4, activation='relu')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='GRU_Simple_Model')
    return model


def create_GRU_model(input_shape, dropout_rate=0.3):
    """
    Create complete GRU model for baseline comparison
    This is the full model including final regression layer
    
    Args:
        input_shape: Input shape (timesteps, features)
        dropout_rate: Dropout rate (default: 0.3)
    
    Returns:
        Complete GRU model for direct prediction
    """
    model = Sequential()
    
    # GRU layers matching the architecture
    model.add(GRU(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(GRU(64, return_sequences=False))
    model.add(Dropout(dropout_rate))
    
    # Final regression layers
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))  # Linear activation for regression
    
    return model


def create_lstm_model(input_shape, dropout_rate=0.3):
    """
    Create LSTM model as alternative to GRU for baseline comparison
    
    Args:
        input_shape: Input shape (timesteps, features)
        dropout_rate: Dropout rate (default: 0.3)
    
    Returns:
        Complete LSTM model for prediction
    """
    model = Sequential()
    
    # LSTM layers
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(dropout_rate))
    
    # Final regression layers
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Linear activation for regression
    
    return model


# Legacy CNN model (kept for compatibility but not used in main pipeline)
def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    """
    Legacy CNN model - kept for compatibility
    Not used in main multimodal pipeline
    """
    inputShape = (height, width, depth)
    chanDim = -1
    
    inputs = Input(shape=inputShape)
    x = inputs
    
    # Loop over filters
    for (i, f) in enumerate(filters):
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Flatten and add dense layers
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    
    x = Dense(4)(x)
    x = Activation("relu")(x)
    
    # Add regression head if requested
    if regress:
        x = Dense(1, activation="linear")(x)
    
    model = Model(inputs, x)
    return model