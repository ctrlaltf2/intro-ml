function [J, grad] = costFunction(theta, X_train, y_train)
    m = size(X_train, 1); % Number of training examples
    n = size(X_train, 2); % Number of features
    
    % Do some assertions to make debugging easier
    assert(all( size(y_train) == [m, 1] ), "y should be a column vector with length equal to number of training examples.");
    assert(all( size(theta)   == [n, 1] ), "theta should be a column vector equal to the number of features in X.");
    
    % Hypotheses
    hypotheses = sigmoid(X_train * theta);
    assert(all( size(hypotheses) == [m, 1] ), "Error getting hypotheses; hypotheses vector didn't match y_train size.");
    
    % -- Cost
    J = (1.0 / m) .* sum(-y_train.*log(hypotheses) - (1-y_train).*log(1 - hypotheses));
    
    % -- Gradient
    h_err = hypotheses - y_train;
    
    % Get the error function's gradient/slope in each dimension (same
    % implementation as linear regression, in part)
    grad = (1.0/m) .* X_train' * h_err;
end

