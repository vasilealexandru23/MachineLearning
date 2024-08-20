function forwardPropagation(w1, b1, w2, b2, w3, b3, x)
    z1 = w1 * x + b1
    a1 = ReLU(z1)

    z2 = w2 * a1 + b2
    a2 = ReLU(z2)

    z3 = w3 * a2 + b3
    a3 = softmax(z3)

    return a3
end