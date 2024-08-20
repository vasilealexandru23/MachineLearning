include("forwardPropagation.jl")
function cost_function(W1, B1, W2, B2, W3, B3, X, Y)
    # Number of training samples
    m = size(X, 1)

    # Number of features
    n = size(X, 2)

    # Compute cost function
    J = 0
    for i ∈ 1:m
        A3 = forwardPropagation(W1, B1, W2, B2, W3, B3, X[i, :])
        J += sum((Y[:, i] - A3).^2)
    end
    J /= 2 * m

    return J
end
