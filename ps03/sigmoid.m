function gz = sigmoid(z)
    gz = 1 ./ (1 + exp(-z));
end

