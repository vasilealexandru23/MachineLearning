using CSV
using DataFrames
using Plots

file = CSV.File("RealEstate.csv"; ignoreemptyrows = true);

fileMatrix = file|>DataFrame|>Matrix;

# Extract number of features
numberFeatures = size(fileMatrix, 2) - 2;

# Extract number of training samples
numberTrainingSamples = size(fileMatrix, 1);

# Ignore first and last column
featureMatrix = fileMatrix[:,2:(numberFeatures + 1)];

# Get system of equations
X = [ones(numberTrainingSamples) featureMatrix];
Y = fileMatrix[:, (numberFeatures + 2)];

# Use normal equation to find theta (Xθ = Y => Xθ = P, P = projection of θ onto the column space of X)
θ = inv(X'X) * (X') * Y;

# Define hypothesis function
h(x) = x' * θ;

# Estimate price of a house
testEntry = X[300,:];
h(testEntry)
