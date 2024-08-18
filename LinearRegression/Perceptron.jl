using CSV
using DataFrames
using Plots
using LinearAlgebra
using Statistics

file = CSV.File("CSVsamples/PerceptronPoints.csv"; ignoreemptyrows = true);

fileMatrix = file|>DataFrame|>Matrix;

numberFeatures = size(fileMatrix, 2) - 1;
numberTrainingSamples = size(fileMatrix, 1);

# Input of training samples
X = fileMatrix[:,2;3]

# Output of training samples
Y = fileMatrix[:,2];



