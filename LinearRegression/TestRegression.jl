using CSV
using DataFrames
using Plots
using LinearAlgebra
using Statistics
include("LocallyWeightedLinearRegression.jl")

file = CSV.File("CSVsamples/PerceptronPoints.csv"; ignoreemptyrows = true);

fileMatrix = file|>DataFrame|>Matrix;

numberFeatures = size(fileMatrix, 2) - 1;
numberTrainingSamples = size(fileMatrix, 1);

# Input of training samples
X = [ones(numberTrainingSamples, 1) fileMatrix[:,1]];

# Output of training samples
Y = fileMatrix[:,2];

P = scatter(X[:,2], Y, seriestype = :scatter, color="green", markersize=5)

while true
    println("Insert total bill to predict tip: ");
    local totalBill = readline();
    totalBill = parse(Float64, totalBill);

    local θ, yP = weightedLR(X, Y, [1.0, totalBill]);
    println(yP);

    h(x) = θ' * x;
    plot!(P, X[:,2], [h(X[i,:]) for i in 1:numberTrainingSamples], color="red", linewidth=4)
    display(P);
end

