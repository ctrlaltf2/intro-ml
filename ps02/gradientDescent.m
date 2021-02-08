function [theta, cost] = gradientDescent(X_train, y_train, alpha, iters)
    % Implementation adapted from Lecture 03, Page 27, Slide 2
    m = size(X_train, 1);
    
    % Start theta off at a random point
    theta = rand(size(X_train, 2), 1);

    % Preallocate cost history vector
    cost = zeros(1, iters);
    
    for i = 1:iters
        % Get the hypotheses and their difference from the ground truth
        % (very similar to computeCost vectorized implementation)
        hypotheses = X_train * theta;
        h_err = hypotheses - y_train;
        
        % Get the error function's gradient/slope in each dimension
        err_slope = (1.0/m) .* X_train' * h_err;
        
        % Update all theta
        theta = theta - alpha .* err_slope;
        
        cost(i) = computeCost(X_train, y_train, theta);
    end
end

