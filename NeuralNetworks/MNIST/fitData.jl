include("ReLU.jl")
include("softmax.jl")
include("cost_function.jl")
include("forwardPropagation.jl")

function fit_data(W1, B1, W2, B2, W3, B3, X, train_y)
    # Number of training samples
    m = size(X, 1)

    # Number of features
    n = size(X, 2)

    # Change y to one-hot encoding
    Y = zeros(10, m)
    for i ∈ 1:m
        Y[Int64(train_y[i]) + 1, i] = 1
    end

    # Declare learning rate
    α = 0.1

    # Save J's value to see convergence
    Jhistory = []

    # Number of iterations
    maxIter = 1000
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
        dW3 = 1 / m * dZ3 * A2'
        dB3 = 1 / m * sum(dZ3)
        dA2 = W3' * dZ3
        dZ2 = dA2 .* (Z2 .> 0)
        dW2 = 1 / m * dZ2 * A1'
        dB2 = 1 / m * sum(dZ2)
        dA1 = W2' * dZ2
        dZ1 = dA1 .* (Z1 .> 0)
        dW1 = 1 / m * dZ1 * A0'
        dB1 = 1 / m * sum(dZ1)

        # Update weights and biases
        W1 .= W1 - α * dW1
        B1 .= B1 .- α * dB1

        W2 .= W2 - α * dW2
        B2 .= B2 .- α * dB2

        W3 .= W3 - α * dW3
        B3 .= B3 .- α * dB3

        # Compute cost function
        Jhistory = [Jhistory; cost_function(W1, B1, W2, B2, W3, B3, X, Y)]
    end

    return Jhistory

end