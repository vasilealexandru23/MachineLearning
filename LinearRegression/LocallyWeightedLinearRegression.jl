# X - training input matrix, Y - output target, xP - to predict
function weightedLR(X, Y, xP)
    # Extract number of features and samples
    numberSamples = size(X, 1);
    numberFeatures = size(X, 2);

    # Define bandwith feature and learning rate
    τ = 0.5;
    α = 0.01;

    # Define weight function 
    w(i) = exp(- (1 / 2) * ((norm(X[i,:] - xP) / τ) ^ 2));

    # Theta params
    global θ = zeros(numberFeatures);

    # Hypothesis
    h(x) = θ' * x;

    # Maximum number of iterations for Gradient Descent
    maxIter = 100;

    # Gradient Descent
    for k ∈ 1:maxIter
        newTheta = copy(θ);
        for j ∈ 1:numberFeatures
            diffSum = 0.0;
            for i ∈ 1:numberSamples
                diffSum += w(i) * (h(X[i,:]) - Y[i]) * X[i,j];
            end
            newTheta[j] = newTheta[j] - α * diffSum;
        end
        global θ = newTheta;
    end

    return [θ, h(xP)]
end
