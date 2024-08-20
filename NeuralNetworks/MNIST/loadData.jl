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
