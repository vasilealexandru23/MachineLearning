using CSV
using DataFrames
using Plots
using LinearAlgebra
using Statistics

file = CSV.File("CSVsamples/CircularPoints.csv"; ignoreemptyrows = true)
fileMatrix = file |> DataFrame |> Matrix

# Number of features
n = size(fileMatrix, 2) - 1

# Number of training samples
m = size(fileMatrix, 1)

# Extract training samples
train_x = fileMatrix[:, 1:n]
train_y = fileMatrix[:, n+1]

# Use feature scaling for train_x (Z score normalization)
for i in 1:n
    μ = mean(train_x[:, i])
    σ = std(train_x[:, i])
    train_x[:, i] = (train_x[:, i] .- μ) ./ σ
end

# Plot the points
plot(train_x[:, 1], train_x[:, 2] , seriestype = :scatter, color = [train_y[i] == 1 ? "green" : "red" for i in 1:m], markersize = 5)

# Define theta params
θ = zeros(n + 1)

# Define sigmoid function
sigmoid(z) = 1 / (1 + exp(-z));

# Define hypothesis function
h(x) = sigmoid(θ[1] + θ[2] * x[2] * x[2] + θ[3] * x[3] * x[3])

θ, J = logisticRegression(train_x, train_y, n, m, h)

# Define decision boundary
dbpos(x) = sqrt((- (θ[1] / θ[3]) - (θ[2] / θ[3]) * x * x))
dbneg(x) = -sqrt((- (θ[1] / θ[3]) - (θ[2] / θ[3]) * x * x))

# Plot the decision boundary
plot!(range(-2, stop = 2, length = 100), dbpos, color = "black", linewidth = 2)
plot!(range(-2, stop = 2, length = 100), dbneg, color = "black", linewidth = 2)

# Plot J history to see convergence of gradient ascent
# plot([J[i][2] for i in 1:size(J, 1)], [J[i][1] for i in 1:size(J, 1)], color = "blue", linewidth = 2, xlabel = "Iterations", ylabel = "Error")