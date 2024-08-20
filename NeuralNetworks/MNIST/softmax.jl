function softmax(z)
    return exp.(z) / sum(exp.(z))
end
