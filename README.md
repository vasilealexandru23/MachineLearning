# Machine Learning models

### This directory contains some machine learning models in Julia & Python3 (TensorFlow).

## `Structure of the directory:`
  * `CSVsamples/` -> Directory with different csv samples data for training LR. 
  * `Clusters/` -> Directory implementing clustering algorithms using k-means.
    * `Data/` -> Directory with all data used for applications.
    * `kmeans.jl` -> Julia implementation for points in 2D plane (3 clusters).
    * `imageCompression.py` -> Python implmenetation of image compression using kmeans (factor of 6).
  * `LinearRegression/` -> Directory with different types of linear regression.
    * `LinearRegression.jl` -> Simple implementation of LR using Normal Equation.
    * `LinearRegression2D.jl` -> A visualization of LR in 2D space (1 feature) using normal equation.
    * `DynamicLinearRegression.jl` -> Implementation for new data coming for trainig using low-rank changes.
    * `LinearRegressionGradientDescent.jl` -> Simple implementation of LR using Gradient Descent and data compression.
    * `LocallyWeightedLinearRegression.jl` -> Linear Regression using locally weighted optimisation for better locally results.
  * `LogisticRegression/` -> Directory with implementation and visualization of logistic regression.
  * `NeuralNetworks/` -> 
    * `MNIST/` -> Two implementations of MNIST in Julia & Python3.
      * `mnist.py` -> Python implementation with one hidden layer with 93% - 97% accuarcy.
      * `mnist.jl` -> Julia implementation using 2 hidden layers with 
      * `visualize.py` -> Tool function to visualize array as image.
    * `CatsDogs/` -> Directory that contains a neural network used to predict if in a JPG image there is a cat or a dog.
  * `SoftmaxRegression/` -> Directory with implementation of softmax regression with 98% accuracy on 3-class dataset.
    * `Softmax.jl` -> The function used for getting the params and cost function history (for convergence test).
    * `TestSoftmax.jl` -> The file were softmax is tested and prints the accuracy and visualizations for J and data points.


**NOTE: For more details about implementation check comments.**

Copyright 2024 Vasile Alexandru-Gabriel (vasilealexandru37@gmail.com)