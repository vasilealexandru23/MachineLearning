function logisticRegression(train_X, Y, n, m, h)
    # Define learning rate
    α = 0.1

    # Add bias term
    X = [ones(m) train_X]
    
    # Theta params
    θ = zeros(n + 1)

    # Define maxIterations
    maxIter = 1000

    # Error function values
    Jhistory = []

    # Apply gradient ascent for maximum likelihood
    for k ∈ 1:maxIter
        newθ = θ
        for j ∈ 1:(n + 1)
            diff = 0.0
            for i ∈ 1:m
                diff += (Y[i] - h(X[i,:])) * X[i, j]
            end

            newθ[j] = newθ[j] + α * diff / m
        end
        θ = newθ

        Jhistory = [Jhistory; (sum([Y[i] * log(h(X[i,:])) + (1 - Y[i]) * log(1 - h(X[i,:])) for i ∈ 1:m]) / m, k)]

    end

    # Return params and error function history
    return θ, Jhistory

end
