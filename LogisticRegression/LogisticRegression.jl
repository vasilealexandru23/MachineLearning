function logisticRegression(train_X, Y, n, m)
    # Define learning rate
    α = 0.01

    # Add bias term
    X = [ones(m) train_X]
    
    # Theta params
    θ = ones(n + 1)

    # Define sigmoid function
    sigmoid(z) = 1 / (1 + exp(-z))

    # Define hypothesis function - change here for different types
    h(x) = sigmoid(θ[1] + θ[2] * x[2] * x[2] + θ[3] * x[3] * x[3])

    # Define maxIterations
    maxIter = 1200

    # Error function values
    Jhistory = []

    # Apply gradient ascent for maximum likelihood
    for k ∈ 1:maxIter
        newθ = θ
        for j ∈ 1:(n + 1)
            grad = 0.0
            for i ∈ 1:m
                grad += (Y[i] - h(X[i,:])) * X[i, j]
            end

            newθ[j] = newθ[j] + (α * grad / m)
        end
        θ = newθ

        Jhistory = [Jhistory; (sum([Y[i] * log(h(X[i,:])) + (1 - Y[i]) * log(1 - h(X[i,:])) for i ∈ 1:m]) / m, k)]

    end

    # Return params and error function history
    return θ, Jhistory

end
