import random
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    img = plt.imread('Data/tiger2.jpg')

    # Divide by 255 so that all pixel values
    # are between 0 and 1 (not needed for PNG files)
    img = img / 255

    return img

def visualize(img):
    plt.imshow(img)
    plt.show()

def prepareImage(img):
    # Reshape the image into an Nx3 matrix where N = number of pixels
    # Each row will contain the Red, Green and Blue for the corresponding pixel

    img = img.reshape(img.shape[0] * img.shape[1], 3)

    return img

def runkMeans(X_img, K, maxIter):
    # Number of pixels
    m = X_img.shape[0]

    # Define bestCost
    bestCost = np.inf

    # Define best cluster centroids for each pixel
    best_idx = np.zeros(m, dtype=int)

    # Define best centroid location
    best_centroids = np.zeros((K, 3))

    for i in range(maxIter):
        # Randomly initalise centroids location
        centroids = np.zeros((K, 3))
        for k in range(K):
            centroids[k,:] = X_img[random.randint(0, m),:]

        # Initialise for each pixel closest centroid
        idx = np.zeros(m, dtype=int)

        # Find for each pixel, closest centroid
        for j in range(m):
            closest = np.inf
            for k in range(K):
                if closest > np.linalg.norm(X_img[j,:] - centroids[k,:]):
                    closest = np.linalg.norm(X_img[j,:] - centroids[k])
                    idx[j] = k
        
        # For each centroid, move it's position
        for k in range(K):
            count = 0
            k_mean = np.zeros(3)

            for j in range(m):
                if idx[j] == k:
                    k_mean += X_img[j,:]
                    count += 1

            if count != 0:
                k_mean /= count
            
            centroids[k,:] = k_mean
            
        # Compute the cost function
        cost = 0
        for j in range(m):
            cost += np.linalg.norm(X_img[j,:] - centroids[idx[j]]) ** 2

        # Update if needed
        cost /= m
        if cost < bestCost:
            bestCost = cost
            best_centroids = centroids
            best_idx = idx

    return best_centroids, best_idx

if __name__ == '__main__':
    img = load_data()

    # Visualize the original image
    visualize(img)

    # Prepare the image for K-means
    X_img = prepareImage(img)

    # Number of clusters and maximum iterations
    K = 16
    maxIter = 10

    # Run K-means algorithm
    centroids, idx = runkMeans(X_img, K, maxIter) 

    # Assign each pixel to the closest centroid
    X_recovered = centroids[idx,:]

    # Reshape the recovered image into proper dimensions
    X_recovered = np.reshape(X_recovered, img.shape)

    # Visualize the compressed image
    visualize(X_recovered)
