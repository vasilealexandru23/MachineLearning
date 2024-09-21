import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import h5py

def load_happy_dataset():
    """
    Loading the dataset and returning the specific arrays:
    
    Arguments:
    None

    Returns:
    X_train, Y_train, X_test, Y_test(training and testing arrays)
    """

    train_dataset = h5py.File('training_samples/train_happy.h5', "r")
    test_dataset = h5py.File('training_samples/test_happy.h5', "r")

    X_train = np.array(train_dataset["train_set_x"][:])
    Y_train = np.array(train_dataset["train_set_y"][:])
    X_test = np.array(test_dataset["test_set_x"][:])
    Y_test = np.array(test_dataset["test_set_y"][:])

    return X_train, Y_train, X_test, Y_test

def happyModel(padding, input_shape, filters, kernel_size, strides, units, activation):
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Arguments:
    padding -> padding param for zero padding operation
    input_shape -> the expected rezolution of images
    filters -> number of filters used in first conv operation
    kernel_size -> size of each filter
    strides -> strides used in conv operation
    units -> output units
    activation -> last activation for output

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    model = tf.keras.Sequential([
        tf.keras.layers.ZeroPadding2D(padding=padding, input_shape=input_shape),
        tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides),
        tf.keras.layers.BatchNormalization(axis=3),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=units, activation=activation)
    ])

    return model

if __name__ == '__main__':
    # Load the dataset
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_happy_dataset()

    # Normalize input
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

    # Reshape output vector
    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    # Visualize data
    index = 1
    plt.imshow(X_train_orig[index])
    plt.show()

    # Create the Sequential Model
    model = happyModel(3, (64,64,3), 32, (7,7), 1, 1, 'sigmoid')

    # Get summary of the model
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train and Evaluate Model
    history = model.fit(X_train, Y_train, epochs=10, batch_size=16, validation_data=(X_test, Y_test))
