using CSV
using DataFrames
using Plots
using LinearAlgebra
using Statistics

file = CSV.File("CSVsamples/Simple.csv"; ignoreemptyrows = true);

fileMatrix = file|>DataFrame|>Matrix;

numberFeatures = size(fileMatrix, 2) - 1;
numberTrainingSamples = size(fileMatrix, 1);

#Maximum number of iteration
maxIter = 100;

# Learning rate
α = 0.01;

# Output of training samples
Y = fileMatrix[:,2];

# Input of training samples
X = [ones(numberTrainingSamples) fileMatrix[:,1]];

# Compress data to converge easily
mean_X = mean(fileMatrix[:,1]);
std_X = std(fileMatrix[:, 1]);
X[:,2] -= mean_X * ones(size(X,1), 1);
X[:,2] /= std_X;

# Theta params
θ = [0.0; 0.0];

h(x) = θ' * x;

for k ∈ 1:maxIter
    newTheta = copy(θ);
    for j ∈ 1:(numberFeatures + 1)
        diffSum = 0.0;
        for i ∈ 1:numberTrainingSamples
            diffSum = diffSum + (h(X[i,:]) - Y[i]) * X[i,j];
        end
        newTheta[j] = newTheta[j] - α * diffSum;
    end
    global θ = newTheta;
end

plot(X[:,2], Y, seriestype = :scatter, color="green", markersize=5)
plot!(X[:,2], [h(X[i,:]) for i in 1:numberTrainingSamples], color="red", linewidth=4)