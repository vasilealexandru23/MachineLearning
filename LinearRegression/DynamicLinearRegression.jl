using CSV
using DataFrames
using Plots
using LinearAlgebra
using Statistics

file = CSV.File("CSVsamples/Simple.csv"; ignoreemptyrows = true);

fileMatrix = file|>DataFrame|>Matrix;

numberFeatures = size(fileMatrix, 2) - 1;
numberTrainingSamples = size(fileMatrix, 1);

X = [ones(numberTrainingSamples) fileMatrix[:,1]]
Y = fileMatrix[:,2]

invData = inv(X'X);

θ = invData * X' * Y;

h(x) = θ' * x;

# Visualization
plot(X[:,2], Y, seriestype = :scatter, color="green", markersize=5)
plot!(X[:,2], [h(X[i,:]) for i in 1:numberTrainingSamples], color="red", linewidth=2)

# Read data

while true
    println("Read SAT: ")
    sat = readline()
    sat = parse(Float64, sat)

    println("Read GPA")
    gpa = readline()
    gpa = parse(Float64, gpa)

    # θ = (invData - invData * newDataEntry * 
    #         inv(eye - sat * invData * sat) * sat * invData)
    #     * (X' * Y + newDataEntry * 
end
