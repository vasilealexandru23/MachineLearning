using CSV
using DataFrames
using Plots
using LinearAlgebra
using Statistics

file = CSV.File("Clusters/data.csv"; ignoreemptyrows = true);

fileMatrix = file|>DataFrame|>Matrix;

# Extract the points
X = fileMatrix[:, 1:end-1];

# Number of points
m = size(X, 1);

# Number of features
n = size(X, 2);

# Do Z-score normalization
for i ∈ 1:n
    X[:, i] = (X[:, i] .- mean(X[:, i])) ./ std(X[:, i]);
end

# Plot the points
plot(X[:, 1], X[:, 2], seriestype = :scatter, title = "Data", label = "Data", xlabel = "X1", ylabel = "X2")

# Number of clusters
K = 3;

# Number of iterations
maxIter = 100;

# Closest centroid for points and best initial cost
best_cost = Inf;
best_c = zeros(m);

for i ∈ 1:maxIter

    local μ = zeros(K, n);
    # Randomly initialise cluster centroids
    for j ∈ 1:K
        μ[j, :] = X[rand(1:m), :];
    end
    
    # Initalize nearly cluster centroid vector
    local c = zeros(m);

    # Find the closest centroid to every point
    for j ∈ 1:m
        closest = Inf  
        for k ∈ 1:K
            if closest > norm(X[j,:] - μ[k,:])
                closest = norm(X[j,:] - μ[k,:])
                c[j] = k
            end
        end
    end

    for k ∈ 1:K
        new_centroid = zeros(n, 1)
        count = 0
        for j ∈ 1:m
            if c[j] == k
                new_centroid += X[j, :]
                count += 1
            end
        end

        if (count == 0)
            μ[k, :] = X[rand(1:m), :]
            continue
        end

        μ[k, :] = new_centroid / count
    end

    # Compute new cost
    J = 1 / m * sum([norm(X[i, :] - μ[Int(c[i]), :])^2 for i ∈ 1:m])

    if J < best_cost
        global best_cost = J
        global best_c = c
    end

end

# Plot the points colored by cluster
plot!(X[best_c .== 1, 1], X[best_c .== 1, 2], seriestype = :scatter, label = "Cluster 1", color = :red)
plot!(X[best_c .== 2, 1], X[best_c .== 2, 2], seriestype = :scatter, label = "Cluster 2", color = :blue)
plot!(X[best_c .== 3, 1], X[best_c .== 3, 2], seriestype = :scatter, label = "Cluster 3", color = :green)