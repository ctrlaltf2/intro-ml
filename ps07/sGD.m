function [Theta1, Theta2] = sGD(input_layer_size,hidden_layer_size, num_labels, X_train, y_train, lambda, alpha, MaxEpochs)
%sGD Perform backpropagation using stochastic gradient descent
    m = size(X_train, 2);

    % Part a)
    % Initialize Theta1 and Theta2 to some values between -0.1 and 0.1
    Theta1 = (2*rand(hidden_layer_size, input_layer_size + 1 ) - 1) / 10;
    Theta2 = (2*rand(num_labels,        hidden_layer_size + 1) - 1) / 10;
    
    % Mini-batching stuff
    % I implemented mini-batching while trying to fix the issue of stopping
    % too early (with the goal of making the descent steps taken less
    % erratic). It turns out this wasn't needed, but I'm leaving it in
    % because it can be configured to play around with mini-batching and
    % full gradient descent. __Currently it is configured for normal sGD
    % which is what this assignment is about.__
    % Idea for this came from 3blue1brown's YouTube series on neural
    % networks.
    mini_batch_size = 1; % Switch to 1 for stochastic. 2 or more for mini-batch. m for normal gradient descent.
    Theta1_d = zeros(size(Theta1));
    Theta2_d = zeros(size(Theta2));
    
    % instructions do say 10^-4 but my sGD was stopping way too early with that, and getting a 33% accuracy.
    convergence_epsilon = 10^-6;
    
    % Convert y_train to one-hot
    y_encoded = double(y_train == unique(y_train)');
    
    J = nnCost(Theta1, Theta2, X_train, y_train, num_labels, lambda);

    J_history = [];
    
    iter = 1;
    for M = [1:MaxEpochs]
        n = 1;
        for x = X_train'
            % Part b)
            y = y_encoded(n, :);
            % Perform a forward pass to grab node values

            % Setup input layer
            a1 = [1; x];

            % Run hidden layer
            z2 = Theta1 * a1;
            a2 = [1; sigmoid(z2)];

            % Run output layer
            z3 = Theta2 * a2;
            a3 = [sigmoid(z3)];

            % Back propagate
            del3 = a3 - y';
            t = Theta2' * del3;
            t = t(2:end, :);
            del2 = t .* sigmoidGradient(z2);

            % Outer product to get the Delta
            Delta2 = del3 * a2';
            Delta1 = del2 * a1';

            % Part c)
            % Get hidden -> output layer's final derivative
            D2 = Delta2;
            D2(:, [2:size(D2, 2)]) = D2(:, [2:size(D2, 2)]) - lambda .* Theta2(:, [2:size(Theta2, 2)]);

            % Get input -> hidden layer's final derivative
            D1 = Delta1;
            D1(:, [2:size(D1, 2)]) = D1(:, [2:size(D1, 2)]) - lambda .* Theta1(:, [2:size(Theta1, 2)]);
            
            % Part d) Update weights
            Theta2_d = Theta2_d + Delta2;
            Theta1_d = Theta1_d + Delta1;

            % Update thetas then reset running delta accumulator
            if mod(iter, mini_batch_size) == 0
                Theta2 = Theta2 - (alpha/mini_batch_size) .* Theta2_d;
                Theta1 = Theta1 - (alpha/mini_batch_size) .* Theta1_d;

                Theta2_d = zeros(size(Theta2));
                Theta1_d = zeros(size(Theta1));
            end
            
            % Part e) Cost
            J_new = nnCost(Theta1, Theta2, X_train, y_train, num_labels, lambda);
            
            dJ = J_new - J;
            if (dJ < 0) && (abs(dJ) < convergence_epsilon)
                % plot([1:length(J_history)], J_history)
                return
            end

            J_history = [J_history J_new];
            J = J_new;

            n = n + 1;
            iter = iter + 1;
        end
    end
    
    % plot([1:length(J_history)], J_history)
end

