function modelCompile(n, m, units1, units2, units3)
    W1 = randn(units1, n) .- 0.5
    B1 = randn(units1) .- 0.5

    W2 = randn(units2, units1) .- 0.5
    B2 = randn(units2) .- 0.5

    W3 = randn(units3, units2) .- 0.5
    B3 = randn(units3) .- 0.5

    return W1, B1, W2, B2, W3, B3
end