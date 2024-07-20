using CSV
using DataFrames
using Plots
using LinearAlgebra
using Statistics

file = CSV.File("Simple.csv"; ignoreemptyrows = true);

fileMatrix = file|>DataFrame|>Matrix;