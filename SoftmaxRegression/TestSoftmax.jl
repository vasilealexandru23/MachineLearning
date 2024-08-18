using CSV
using DataFrames
using Plots
using LinearAlgebra
using Statistics

file = CSV.File("CSVsamples/Iris.csv"; ignoreemptyrows = true)
fileMatrix = file |> DataFrame |> Matrix

# Number of features - ignore id,class,species
n = size(fileMatrix, 2) - 3

# Number of training samples
m = size(fileMatrix, 1)

# Extract data
train_x = fileMatrix[:,2:5]
train_y = fileMatrix[:,6]

# Plot data
# plot(train_x[:,1], train_x[:,4], seriestype = :scatter,
     # color=color = [train_y[i] == 1 ? "green" : (train_y[i] == 2 ? "blue" : "red") for i in 1:m])

θ, J = softmax(train_x, train_y, n, m, 3)

# Plot J history to see convergence of gradient descent
plot([J[i][1] for i in 1:size(J, 1)], [J[i][2] for i in 1:size(J, 1)], color = "blue", linewidth = 2, xlabel = "Iterations", ylabel = "Error")

# Predict
function predict(x, θ)
    X = [1; x]
    return argmax([exp(θ[j,:]' * X) / (sum([exp(θ[i,:]' * X) for i in 1:3])) for j in 1:3])
end

# Get accuracy
function accuracy(x, y, θ)
    return sum([predict(x[i, :], θ) == y[i] for i in 1:size(x, 1)]) / size(x, 1)
end

println("Accuracy: ", accuracy(train_x, train_y, θ))