function [theta] = Reg_normalEqn(X_train, y_train, lambda)
    % Assertions / assumptions
    m = size(X_train, 1);
    n = size(X_train, 2);
    assert( all(size(y_train) == [m 1] ), "y_train didn't match X_train's number of training examples.");
    
    holey_ident = eye(n);
    holey_ident(1, 1) = 0;
    
    theta = pinv(X_train' * X_train + lambda*holey_ident) * X_train' * y_train;
end

