function J = computeCost(X, y, theta)    
    m = size(X, 1);
    
    % First get all the hypotheses
    hypotheses = X * theta;
    
    % Then do the mean squared sum of the difference in hypothesis and
    % actual
    h_err = hypotheses - y;
    J = (1/(2*m)) * (h_err') * h_err;
end