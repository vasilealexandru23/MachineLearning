import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

def load_signs_dataset():
    """
    Loading the dataset and returning the specific arrays:
    
    Arguments:
    None

    Returns:
    X_train, Y_train, X_test, Y_test(training and testing arrays)
    """

    train_dataset = h5py.File('training_samples/train_signs.h5', 'r')
    test_dataset = h5py.File('training_samples/test_signs.h5', 'r')

    X_train = np.array(train_dataset["train_set_x"][:])
    Y_train = np.array(train_dataset["train_set_y"][:])

    X_test = np.array(test_dataset["test_set_x"][:])
    Y_test = np.array(test_dataset["test_set_y"][:])

    return X_train, Y_train, X_test, Y_test

def convert_to_one_hot(y):
    """
    Transforming vector into one-hot encoding vector.

    Arguments:
    y - vector to transform

    Returns:
    One-hot encoding vector
    """

    y_one_hot = np.zeros((y.size, y.max() + 1))
    y_one_hot[np.arange(y.size), y] = 1
    
    return y_one_hot

def signsModel(input_shape):
    """
        Implements the forward propagation for the model:
        CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
        Arguments:
        input_img -- input dataset, of shape (input_shape)

        Returns:
        model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    Z1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(4,4), strides=1, padding='same')(input_img)
    A1 = tf.keras.layers.ReLU()(Z1)
    P1 = tf.keras.layers.MaxPool2D(pool_size=(8,8), strides=8, padding='same')(A1)
    Z2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(2,2), strides=1, padding='same')(P1)
    A2 = tf.keras.layers.ReLU()(Z2)
    P2 = tf.keras.layers.MaxPool2D(pool_size=(4,4), strides=4, padding='same')(A2)
    F = tf.keras.layers.Flatten()(P2)
    outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)

    return model

if __name__ == '__main__':
    # Extract data from datasets
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_signs_dataset()

    # Visualize data
    # index = 224
    # plt.imshow(X_train_orig[index])
    # plt.show()

    # Modify data to specific ones
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    Y_train = convert_to_one_hot(Y_train_orig.T)
    Y_test = convert_to_one_hot(Y_test_orig.T)

    # Build model
    model = signsModel((64,64,3))

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Get the summary
    model.summary()

    # Train and Evaluate
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
    history = model.fit(train_dataset, epochs=200, validation_data=test_dataset)

    # The history.history["loss"] entry is a dictionary with as many values as epochs that the
    # model was trained on. 
    df_loss_acc = pd.DataFrame(history.history)
    df_loss= df_loss_acc[['loss','val_loss']]
    df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
    df_acc= df_loss_acc[['accuracy','val_accuracy']]
    df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
    df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
    df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
    plt.show()
