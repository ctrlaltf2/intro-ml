function [theta] = normalEqn(X_train, y_train)
    % From one of the recent lectures
    theta = pinv(X_train' * X_train) * X_train' * y_train;
end

