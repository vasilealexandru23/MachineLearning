# Training samples, # features, # samples, # classes of y
function softmax(train_x, train_y, n, m, N)
    # Set learning rate
    α = 0.01

    # Modelate data
    X = [ones(m) train_x]
    Y = train_y

    # Define params
    θ = zeros(N, n + 1)

    # Define function
    a(j, x) = exp(θ[j,:]' * x) / (sum([exp(θ[i,:]' * x) for i in 1:N]))

    # Define maximum iterations
    maxIter = 2500

    # Keep history to see convergence
    Jhistory = []

    for k ∈ 1:maxIter
        newθ = θ

        for j ∈ 1:N
            grad = zeros(n + 1)

            for i ∈ 1:m
                grad += (a(j, X[i, :]) - (Y[i] == j ? 1 : 0)) * X[i, :]
            end

            newθ[j, :] = newθ[j, :] - (α * grad / m)
        end

        Jhistory = [Jhistory; (k, (-1 / m) * sum([Y[i] == j ? log(a(j, X[i, :])) : 0 for i in 1:m, j in 1:N]))]
    end

    return θ, Jhistory
end