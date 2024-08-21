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

    return train_x, train_y
end

function modelCompile(n, units1, units2, units3)
    W1 = randn(units1, n) .- 0.5
    B1 = randn(units1) .- 0.5

    W2 = randn(units2, units1) .- 0.5
    B2 = randn(units2) .- 0.5

    W3 = randn(units3, units2) .- 0.5
    B3 = randn(units3) .- 0.5

    return W1, B1, W2, B2, W3, B3
end

function ReLU(Z)
    return max.(0, Z)
end

function softmax(z)
    return exp.(z) / sum(exp.(z))
end


function forwardPropagation(w1, b1, w2, b2, w3, b3, x)
    z1 = w1 * x + b1
    a1 = ReLU(z1)

    z2 = w2 * a1 + b2
    a2 = ReLU(z2)

    z3 = w3 * a2 + b3
    a3 = softmax(z3)

    return a3
end

function cost_function(W1, B1, W2, B2, W3, B3, X, Y)
    # Number of training samples
    m = size(X, 1)

    # Compute cost function
    J = 0
    for i ∈ 1:m
        A3 = forwardPropagation(W1, B1, W2, B2, W3, B3, X[i, :])
        J += sum((Y[:, i] - A3) .^ 2)
    end
    J /= 2 * m

    return J
end

function fit_data(W1, B1, W2, B2, W3, B3, X, train_y)
    # Number of training samples
    m = size(X, 1)

    # Number of features
    n = size(X, 2)

    # Change y to one-hot encoding
    Y = zeros(10, m)
    for i ∈ 1:m
        Y[Int64(train_y[i])+1, i] = 1
    end

    # Declare learning rate
    α = 0.1

    # Save J's value to see convergence
    Jhistory = []

    # Number of iterations
    maxIter = 200
    for k ∈ 1:maxIter
        # Forward propagation
        A0 = X'
        Z1 = W1 * A0 .+ B1
        A1 = ReLU(Z1)
        Z2 = W2 * A1 .+ B2
        A2 = ReLU(Z2)
        Z3 = W3 * A2 .+ B3
        A3 = softmax(Z3)

        # Backward propagation
        dZ3 = A3 - Y
        dW3 = dZ3 * A2'
        dB3 = sum(dZ3)
        dA2 = W3' * dZ3
        dZ2 = dA2 .* (Z2 .> 0)
        dW2 = dZ2 * A1'
        dB2 = sum(dZ2)
        dA1 = W2' * dZ2
        dZ1 = dA1 .* (Z1 .> 0)
        dW1 = dZ1 * A0'
        dB1 = sum(dZ1)

        # Update weights and biases
        W1 .= W1 - α / m * dW1
        B1 .= B1 .- α / m * dB1

        W2 .= W2 - α / m * dW2
        B2 .= B2 .- α / m * dB2

        W3 .= W3 - α / m * dW3
        B3 .= B3 .- α / m * dB3

        # Compute cost function
        Jhistory = [Jhistory; cost_function(W1, B1, W2, B2, W3, B3, X, Y)]
    end

    return Jhistory
end

function main()
    # Load samples
    train_x, train_y = load_data()

    # Number of features -> pixels of image
    n = size(train_x, 2)

    # Number of training samples
    m = size(train_x, 1)

    # Prepare the neural network
    W1, B1, W2, B2, W3, B3 = modelCompile(n, 10, 10, 10)

    # Fit params to minimise cost function (use least squares, gradient descent)
    Jhistory = fit_data(W1, B1, W2, B2, W3, B3, train_x, train_y)

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

end

main()