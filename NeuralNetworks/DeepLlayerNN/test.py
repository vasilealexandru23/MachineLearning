import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from LlayerNN import NeuralNetwork
from PIL import Image

def load_input():
    # Load all images from train folder
    # Prepare X_train
    X_train = np.zeros((400, 64 * 64 * 3))
    for i in range(1, 200):
        img = tf.keras.utils.load_img(
            '../../training_set_big/cats/cat.' + str(i) + '.jpg',
            target_size=(64,64),
            color_mode='rgb',
            interpolation='nearest')

        img = tf.keras.utils.img_to_array(img)
        img = np.array(img)/255.

        # Reshape image to a column vector
        img = img.reshape(img.shape[0] * img.shape[1] * img.shape[2], 1)

        X_train[i, :, None] = img

    for i in range(1, 200):
        img = tf.keras.utils.load_img(
            '../../training_set_big/dogs/dog.' + str(i) + '.jpg',
            target_size=(64,64),
            color_mode='rgb',
            interpolation='nearest')

        img = tf.keras.utils.img_to_array(img)
        img = np.array(img)/255. 

        # Reshape image to a column vector
        img = img.reshape(img.shape[0] * img.shape[1] * img.shape[2], 1)

        X_train[i + 200, :, None] = img

    return X_train

def load_output():
    # Load all images from train folder
    # Prepare Y_train
    Y_train = np.zeros((400, 1))

    for i in range(200, 400):
        Y_train[i] = 1

    return Y_train

def shuffle(X_train, Y_train):
    T = np.c_[X_train, Y_train]

    np.random.shuffle(T)

    X = T[:,0:-1]
    Y = T[:,-1, None].astype(int)

    return X.T, Y.T

def main():
    # Load all images from train folder and prepare X_train and Y_train
    X_train = load_input()
    Y_train = load_output()

    # Number of training samples
    m = X_train.shape[0]

    # Shuffle data
    X, Y = shuffle(X_train, Y_train)

    # Instance the model
    model = NeuralNetwork([64 * 64 * 3, 128, 64, 64, 1], ["relu", "relu", "relu", "sigmoid"])

    # Compile model, initialize params
    model.compile(0.001, 100)
    
    # Run gradient descent to fit the model
    J = model.fit(X, Y, optimizer="adam")

    # Plot cost function history
    J = np.array(J)
    plt.plot(J[:,0], J[:,1])
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.show()

    # Get accuracy over all trainig set
    correct = 0
    for i in range(m):
        if model.predict(X[:,i, None]) == Y[0, i]:
            correct += 1
    
    print("Accuracy: ", correct / m)

if __name__ == '__main__':
    main()
