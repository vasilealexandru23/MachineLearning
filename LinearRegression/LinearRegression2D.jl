using CSV
using DataFrames
using Plots

file = CSV.File("CSVsamples/2Dim.csv"; ignoreemptyrows = true);

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

# Use normal equation to find theta
θ = inv(X'X) * (X') * Y;

# Define hypothesis function
h(x) = θ' * x;

####### Visualization #######

plot(X[:,2], Y, seriestype = :scatter, color="green", markersize=5)
plot!(X[:,2], [h(X[i,:]) for i in 1:numberTrainingSamples], color="red", linewidth=2)
