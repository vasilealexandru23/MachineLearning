import numpy as np
import tensorflow as tf

# Function to create an identity block ("skip connection = 3 layers")
# Attention: the shape of the initial input is the same as the shape of the output
def identity_block(X, f, filters, initializer=tf.keras.initializers.RandomUniform):
    """
    Implementation of the identity block 
    X -> Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> Add -> ReLU
    ↓ -----------------------------------------------------------------------------------------↑

    Arguments:
    X -- input tensor of shape (m, n_H, n_W, n_C)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- number of filters used in the CONV layers
    initializer -- initializer for the weights (default: random_uniform)

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # Retrieve filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = tf.keras.layers.Conv2D(filters = F1, kernel_size = 1, strides = (1, 1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.ReLU()(X)

    # Second component of main path
    X = tf.keras.layers.Conv2D(filters = F2, kernel_size = f, strides = (1, 1), padding = 'same', kernel_initializer = initializer(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.ReLU()(X)

    # Third component of main path
    X = tf.keras.layers.Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)

    # Add the shortcut value to the final component
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.ReLU()(X)

    return X

# Function to create an convolutional block ("skip connection = 3 layers")
# There is an extra conv layer for the input the transform to output shape
def convolutional_block(X, f, filters, s = 2, initializer = tf.keras.initializers.GlorotUniform):
    """
    Implementation of the convolution block
    X -> Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> Add -> ReLU
    ↓ -------------------------------> Conv2D -> BatchNorm -> ---------------------------------↑

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying teh hsape of the middle CONV's window for the main path
    filters -- number of filters used in the CONV layers
    s -- the stride used in layers (default = 2)
    initializer -- initializer for the weights (default : glorot_uniform)

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # Retrieve filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of the main path
    X = tf.keras.layers.Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.ReLU()(X)

    # Second component of the main path
    X = tf.keras.layers.Conv2D(filters = F2, kernel_size = f, strides = (1, 1), padding = 'same', kernel_initializer = initializer(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)
    X = tf.keras.layers.ReLU()(X)

    # Third component of the main path
    X = tf.keras.layers.Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X)

    # Shortcut path
    X_shortcut = tf.keras.layers.Conv2D(filters = F3, kernel_size = 1, strides = (s, s), padding = 'valid', kernel_initializer = initializer(seed=0))(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization(axis = 3)(X_shortcut)

    # Add the shortcut value to the last component of the main path
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.ReLU()(X)

    return X
