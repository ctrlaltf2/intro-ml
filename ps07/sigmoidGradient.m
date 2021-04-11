function g_prime = sigmoidGradient(z)
    g_prime = sigmoid(z) .* (1 - sigmoid(z));
end

