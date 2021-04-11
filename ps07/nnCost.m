function J = nnCost(Theta1, Theta2, X, y, K, lambda)
    m = size(X, 1);
    
    [y_pred, a_out] = predict(Theta1, Theta2, X);

    % one-hot encode y e.g. 2 -> [0 1 0], 3 -> [0 0 1], etc.
    labels = unique(y)';
    y      = (y      == labels);
    y_pred = double(y_pred == labels);
    
    % use the raw output values instead of the prediction labels
    % Using the prediction labels was giving me NaN and -Inf (from the
    % exact 0s and 1s). Also, raw 0s/1s might optimize for the neural
    % network to train for things like [0.4599 0.46 0.4599] when [0 1 0] is
    % correct; we want something like [0.01 0.97 0.02].
    h_x = a_out;
    
    K_loop = sum(y.*log(h_x) + (1 - y).*log(1 - h_x), 2);
    m_loop = sum(K_loop, 1);
    
    % sum every single theta^2 but leave out bias thetas
    reg = (lambda / (2*m)) * (sum(sum(Theta1(:, [2:size(Theta1, 2)]).^2)) + sum(sum(Theta2(:, [2:size(Theta2, 2)]).^2)));
    
    J = -(1/m) * m_loop + reg;
end