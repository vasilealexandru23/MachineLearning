import h5py
import image
import numpy as np
import tensorflow as tf
import residualnetwork
from tensorflow.keras.preprocessing import image
from matplotlib.pyplot import imshow

def load_signs_dataset():
    """
    Loading the dataset and returning the specific arrays:
    
    Arguments:
    None

    Returns:
    X_train, Y_train, X_test, Y_test(training and testing arrays)
    """

    train_dataset = h5py.File('../CNN/training_samples/train_signs.h5', 'r')
    test_dataset = h5py.File('../CNN/training_samples/test_signs.h5', 'r')

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

def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    X_input = tf.keras.layers.Input(input_shape)

    # Zero-Padding
    X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = tf.keras.layers.Conv2D(filters = 64, kernel_size = 7, strides = (2, 2), padding = 'valid', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis = 3)(X) 
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides = (2, 2))(X)

    # Stage 2
    X = residualnetwork.convolutional_block(X, 3, [64, 64, 256], s = 1)
    X = residualnetwork.identity_block(X, 3, [64, 64, 256])
    X = residualnetwork.identity_block(X, 3, [64, 64, 256])

    # Stage 3
    X = residualnetwork.convolutional_block(X, 3, [128, 128, 512])
    X = residualnetwork.identity_block(X, 3, [128, 128, 512])
    X = residualnetwork.identity_block(X, 3, [128, 128, 512])
    X = residualnetwork.identity_block(X, 3, [128, 128, 512])

    # Stage 4
    X = residualnetwork.convolutional_block(X, 3, [256, 256, 1024])
    X = residualnetwork.identity_block(X, 3, [256, 256, 1024])
    X = residualnetwork.identity_block(X, 3, [256, 256, 1024])
    X = residualnetwork.identity_block(X, 3, [256, 256, 1024])
    X = residualnetwork.identity_block(X, 3, [256, 256, 1024])
    X = residualnetwork.identity_block(X, 3, [256, 256, 1024])

    # Stage 5
    X = residualnetwork.convolutional_block(X, 3, [512, 512, 2048])
    X = residualnetwork.identity_block(X, 3, [512, 512, 2048])
    X = residualnetwork.identity_block(X, 3, [512, 512, 2048])

    X = tf.keras.layers.AveragePooling2D((2, 2))(X)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(classes, activation='softmax', kernel_initializer = tf.keras.initializers.glorot_uniform(seed=0))(X)

    model = tf.keras.Model(inputs = X_input, outputs = X)
    return model

if __name__ == '__main__':
    # Extract data from datasets
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_signs_dataset()

    # Visualize data
    # index = 225
    # plt.imshow(X_train_orig[index])
    # plt.show()

    # Modify data to specific ones
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    Y_train = convert_to_one_hot(Y_train_orig.T)
    Y_test = convert_to_one_hot(Y_test_orig.T)

    # Build model
    model = ResNet50(input_shape = (64, 64, 3), classes = 6)

    # Compile and evaluate model
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00015), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    history = model.fit(X_train, Y_train, epochs=10, batch_size=16, validation_data=(X_test, Y_test))

    # Test on my images
    img = image.load_img('3.jpg', target_size=(64, 64))
    x_test = image.img_to_array(img)
    x_test = np.expand_dims(x_test, axis=0)
    x_test = x_test/255.
    imshow(img)
    prediction = model.predict(x_test)
    print("Class:", np.argmax(prediction))
