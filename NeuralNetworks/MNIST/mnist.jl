using CSV
using DataFrames
using Plots
using LinearAlgebra
using Statistics

include("fitData.jl")
include("loadData.jl")
include("modelCompile.jl")

# Load samples
train_x, train_y = load_data()

# Number of features -> pixels of image
n = size(train_x, 2)

# Number of training samples
m = size(train_x, 1)

# Prepare the neural network
W1, B1, W2, B2, W3, B3 = modelCompile(n, m, 16, 16, 10)

# Fit params to minimise cost function (use least squares, gradient descent)
Jhistory = fit_data(W1, B1, W2, B2, W3, B3, train_x, train_y);

# Plot the cost function
plot(Jhistory, title="Cost function", label="J", xlabel="Iteration", ylabel="J", lw=2)

# Predict
function predict(x)
    A3 = forwardPropagation(W1, B1, W2, B2, W3, B3, x)
    return argmax(A3) - 1
end

# Get accuracy for the model
global correct = 0
for i ∈ 1:m
    if predict(train_x[i, :]) == train_y[i]
        global correct += 1
    end
end
print(correct / m)