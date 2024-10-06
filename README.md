# Machine Learning models

### This directory contains some machine learning models in Julia & Python3 (TensorFlow).

## `Structure of the directory:`
  * `CSVsamples/` -> Directory with different csv samples data for training LR. 
  * `FaceRecognition/` -> Directory with implementation of Face Verification and Recognition using pretrained Inception Model.
  * `ImageSegmentation/` -> Directory with implmentation of U-Net used in Image Segmentation.
  * `LinearRegression/` -> Directory with different types of linear regression.
    * `LinearRegression.jl` -> Simple implementation of LR using Normal Equation.
    * `LinearRegression2D.jl` -> A visualization of LR in 2D space (1 feature) using normal equation.
    * `DynamicLinearRegression.jl` -> Implementation for new data coming for trainig using low-rank changes.
    * `LinearRegressionGradientDescent.jl` -> Simple implementation of LR using Gradient Descent and data compression.
    * `LocallyWeightedLinearRegression.jl` -> Linear Regression using locally weighted optimization for better locally results.
  * `LogisticRegression/` -> Directory with implementation and visualization of logistic regression.
  * `NeuralNetworks/` -> Directory with different Neural Networks models and architectures.
    * `CatsDogs/` -> Directory that contains a neural network used to predict cats&dogs small dataset.
    * `CNN/` ->  Directory with implementations of CNN (TensorFlow + scratch) used on two datasets with LeNet-5 Architecture.
    * `DeepLlayerNN/` -> Directory with generic implementation of a deep L layer neural network with different optimizations(mini-batches, momentum, adam) + vizualizations.
    * `MNIST/` -> Two implementations of MNIST in Julia & Python3 with 93%-97% accuracy.
    * `ResNet/` -> Implementation + tests on Residual Neural Network using 50 layers with 80% accuracy on testset.
    * `RNN/` -> Directory with implementation and different applications of Recurent Neural Networks.
  * `Paperworks/` -> Directory with paperworks downloaded from https://arxiv.org/.
  * `SoftmaxRegression/` -> Directory with implementation of softmax regression with 98% accuracy on 3-class dataset.
    * `Softmax.jl` -> The function used for getting the params and cost function history (for convergence test).
    * `TestSoftmax.jl` -> The file where softmax is tested and prints the accuracy and visualizations for J and data points.
  * `StlyeTransfer/` -> Directory with implementation of Art Generation with Neural Style Transfer using a pretrained VGG-19 model.
  * `UnsupervisedLearning/` -> Directory with recommenders and reinforcement learning.
    * `AnomalyDetection` -> Directory with implementation of basic anomaly detection.
    * `Clusters/` -> Directory implementing clustering algorithms using k-means + image compression.
      * `Data/` -> Directory with all data used for applications.
      * `kmeans.jl` -> Julia implementation for points in 2D plane (3 clusters).
      * `imageCompression.py` -> Python implmenetation of image compression using kmeans (factor of 6).
    * `ContentBased/` -> Directory with implementation of content based filtering using 2 neural networks w/ TensorFlow.
    

**NOTE: For more details about implementation check comments.**      
**NOTE: Part of datasets/models/functions used are from [DeepLearning.AI](https://www.deeplearning.ai/).**

Copyright 2024 Vasile Alexandru-Gabriel (vasilealexandru37@gmail.com)
