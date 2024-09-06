using CSV
using DataFrames
using Plots
using LinearAlgebra
using Statistics
using Random

function load_data()
    # Read X trainig
    file = CSV.File("NeuralNetworks/MNIST/TrainX.csv"; ignoreemptyrows=true, header=0)
    train_x = file |> DataFrame |> Matrix

    # Read Y training
    file = CSV.File("NeuralNetworks/MNIST/TrainY.csv"; ignoreemptyrows=true, header=0)
    train_y = file |> DataFrame |> Matrix

    # Combine them into a single Matrix
    Train = [train_x train_y]

    # Shuffle the data
    Train = Train[shuffle(1:end), :]

    # Split the data into X and Y
    train_x = Train[:, 1:end-1]
    train_y = Train[:, end]

    return train_x', train_y'
end

function one_hot(Y, classes)
    # Number of tests
    m = size(Y, 2)

    one_hot_Y = zeros(Int8, classes, m)
    for i ∈ 1:m
        one_hot_Y[(Int8(Y[i]) + 1), i] = 1
    end

    return one_hot_Y
end

function model_compile(features, units1, units2, classes)
    # Initialize parameters
    W1 = randn(units1, features) * (1 / sqrt(features))
    b1 = zeros(units1, 1) * (1 / sqrt(features))
    W2 = randn(units2, units1) * (1 / sqrt(units1))
    b2 = zeros(units2, 1) * (1 / sqrt(units1))
    W3 = randn(classes, units2) * (1 / sqrt(units2))
    b3 = zeros(classes, 1) * (1 / sqrt(units2))

    return W1, b1, W2, b2, W3, b3
end

function ReLU(Z)
    return max.(0, Z)
end

function ReLU_derivative(Z)
    return Z .> 0
end

function softmax(Z)
    return exp.(Z) ./ sum(exp.(Z))
end

function forwardPropagation(X, W1, b1, W2, b2, W3, b3)
    # First layer
    Z1 = W1 * X .+ b1
    A1 = ReLU(Z1)

    # Second layer
    Z2 = W2 * A1 .+ b2
    A2 = ReLU(Z2)

    # Third layer
    Z3 = W3 * A2 .+ b3
    A3 = softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3
end

function computeCost(W1, b1, W2, b2, W3, b3, X_train, one_hot_Y)
    # Number of training samples
    m = size(X_train, 2)

    J = 0

    for i ∈ 1:m
        _, _, _, _, _, A3 = forwardPropagation(X_train[:, i], W1, b1, W2, b2, W3, b3)
        J += -sum(one_hot_Y[:, i] .* log.(A3) + (1 .- one_hot_Y[:, i]) .* log.(1 .- A3))
    end

    return J / m
end

function fit_model(X_train, Y_train, α, maxIter)
    # Number of training samples
    m = size(X_train, 2)

    # Number of pixels, features
    n = size(X_train, 1) 

    # Initialize parameters
    W1, b1, W2, b2, W3, b3 = model_compile(n, 16, 16, 10)

    # One hot encode the labels
    one_hot_Y = one_hot(Y_train, 10)

    # Cost function
    J = []

    for k ∈ 1:maxIter
        # Forward Propagation
        Z1, A1, Z2, A2, Z3, A3 = forwardPropagation(X_train, W1, b1, W2, b2, W3, b3)

        # Backward Propagation
        dZ3 = A3 - one_hot_Y
        dW3 = (1 / m) * dZ3 * A2'
        db3 = (1 / m) * sum(dZ3)

        dZ2 = W3' * dZ3 .* ReLU_derivative(Z2)
        dW2 = (1 / m) * dZ2 * A1'
        db2 = (1 / m) * sum(dZ2)

        dZ1 = W2' * dZ2 .* ReLU_derivative(Z1)
        dW1 = (1 / m) * dZ1 * X_train'
        db1 = (1 / m) * sum(dZ1)

        # Update parameters
        W1 = W1 - α * dW1
        b1 = b1 .- α * db1
        W2 = W2 - α * dW2
        b2 = b2 .- α * db2
        W3 = W3 - α * dW3
        b3 = b3 .- α * db3

        # Compute the cost function
        J = [J; ((k, computeCost(W1, b1, W2, b2, W3, b3, X_train, one_hot_Y)))]
    end

    return J, W1, b1, W2, b2, W3, b3
end

function main()
    # Load samples
    X_train, Y_train = load_data()

    # Number of training samples
    m = size(X_train, 2)

    # Number of pixels, features
    n = size(X_train, 1)

    # Fit data using gradient descent given learning rate and epochs
    J, W1, b1, W2, b2, W3, b3 = fit_model(X_train, Y_train, 0.1, 200)

    # Plot the cost function
    plot([x[1] for x ∈ J], [x[2] for x ∈ J], title="Cost Function", label="Cost Function", xlabel="Epochs", ylabel="Cost")

    # Get the accuracy of the model
    correct = 0
    for i ∈ 1:m
        _, _, _, _, _, A3 = forwardPropagation(X_train[:, i:i], W1, b1, W2, b2, W3, b3)
        if argmax(A3) == Y_train[i]
            correct += 1
        end
    end

    accuracy = correct / m
    println("Accuracy: ", accuracy)

end

main()