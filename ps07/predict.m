function [p, scores] = predict(Theta1, Theta2, X)
    % Preallocate
    p = zeros(size(X, 1), 1);
    scores = zeros(size(X, 1), size(Theta2, 1));

    n = 1;
    for a1 = X'
        % Setup input layer
        a1 = [1; a1];
        
        % Run hidden layer
        z2 = Theta1 * a1;
        a2 = [1; sigmoid(z2)];

        % Run output layer
        z3 = Theta2 * a2;
        a3 = [sigmoid(z3)];

        % Get the prediction (index of the highest node)
        pred = find(a3 == max(a3));
        
        % Save result
        p(n, 1) = pred;
        scores(n, :) = a3';
        
        n = n + 1;
    end
end

