import numpy as np
import matplotlib.pyplot as plt

def load_data():
    data = np.loadtxt(open("creditcard.csv", "r"), delimiter=",", skiprows=1)

    # Shuffle rows
    np.random.shuffle(data)

    # Extract final data
    X = data[:,1:-1]
    Y = data[:,-1, None].astype(int)

    return X.T, Y

def findDistribution(X):
    # Compute mean and standard deviation
    mean = np.mean(X)
    std = np.std(X)

    # Get the gaussian distribution
    def p(x):
        return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    return p

def vizualizeFeatureDistribution(X):
    # Find the distribution for vector -> Try feature engineering
    p = findDistribution(X)
    
    # Plot the function p(x) and the histogram of the data as probability density
    plt.hist(X, bins=100, density=True)
    plt.plot(np.linspace(min(X), max(X), 1000), p(np.linspace(min(X), max(X), 1000)), color='r')
    plt.show()

if __name__ == '__main__':
    # Load data
    X, Y = load_data()

    # Number of features
    n = X.shape[0]

    # Number of samples
    m = X.shape[1]
    
    # Find distribution of each feature
    distrib = []
    for i in range(X.shape[0]):
        distrib.append(findDistribution(X[i, :, None]))

    threshold = 1e-23

    # Test accuracy for the model
    correct_0 = 0
    correct_1 = 0
    all_0 = 0
    all_1 = 0

    for t in range(m):
        pred = 1.0
        for f in range(n):
            curr_distrib = distrib[f]
            pred = pred * curr_distrib(X[f, t, None])

        if pred < threshold:
            pred = 1
        else:
            pred = 0

        if Y[t, 0, None] == 0:
            all_0 += 1
        else:
            all_1 += 1

        if pred == Y[t, 0, None]:
            if Y[t, 0, None] == 0:
                correct_0 += 1
            else:
                correct_1 += 1

    print(correct_0, all_0, correct_1, all_1)
    print((correct_0 + correct_1) / m)
