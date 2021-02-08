%% *Homework Assignment 2: Linear Regression*
% *Due Sunday, February 7th, 2021 at 11:59 pm*
% 
%% *Question 1*
% *Note: See computeCost.m for computeCost definition.*

toy.x = [1 2 3 4]';
toy.y = [2 4 6 8]';

% Utility lambda function to leftpad a vector with 1's
onepad = @(x) ( [ones(length(x), 1), x] );

fprintf("Cost, theta = [0 0.5]': %.4f\n", ...
    computeCost(onepad(toy.x), toy.y, [0 0.5]') ...
);

fprintf("Cost, theta = [3.5 0]': %.4f\n", ...
    computeCost(onepad(toy.x), toy.y, [3.5 0]') ...
);
%% Question 2
% *Note: See gradientDescent.m for gradientDescent defintion.*

alpha = 0.01;
iter = 15;

[theta, cost] = gradientDescent(onepad(toy.x), toy.y, alpha, iter);

fprintf( ...
    "After %d iterations with alpha = %.3f:\ntheta = [%.4f; %.4f]\ncost = %.4f.\n", ...
    iter, alpha, theta(1), theta(2), cost(iter) ...
);
%% Question 3
% *Note: See normalEqn.m for normalEqn definition.*

theta = normalEqn(onepad(toy.x), toy.y);

fprintf("Estimate of theta using normal equation: [%.4f, %.4f]\n", theta(1), theta(2));
%% Question 4
%% 
% *Part a) Loading the data*

% Import data
q4_data.mat = importdata('input/hw2_data1.txt', ',', 0);
q4_data.pop = q4_data.mat(:, 1);
q4_data.profit = q4_data.mat(:, 2); 

%% 
% *Part b) Plotting the data*

% Plot pop vs. profit
plot(q4_data.pop, q4_data.profit, 'rx');
ylabel('Profit in $10,000s');
xlabel('Population of City in 10,000s');

% Save as output png
exportgraphics(gcf, 'output/ps2-4-b.png','Resolution', 200);
%% 
% *Part c) Define X and y*

q4_X = onepad(q4_data.pop); % See Q1 for onepad definition
q4_y = q4_data.profit;

fprintf(...
    "The size of X is %dx%d\nThe size of y is %dx%d\n", ...
    size(q4_X, 1), size(q4_X, 2), size(q4_y, 1), size(q4_y, 2) ...
);

%% 
% *Part d) Test & Training set division*

M = size(q4_y, 1);   % Total number of samples
T = floor(0.90 * M); % Number of samples to use for training (~90%)

% Generate indices such that trn_idx randomly points to ~90% of the data,
% and tst_idx to the remaining ~10% of the data
trn_idx = randperm(M, T);          % Training data indices
tst_idx = setdiff([1:M], trn_idx); % Testing data indices (setwise diff [1 ... M] - trn_idx)

% Randomly split up the data based on the generated indices
X_train = q4_X(trn_idx, :);
y_train = q4_y(trn_idx, :);

X_test = q4_X(tst_idx, :);
y_test = q4_y(tst_idx, :);

% Note: if there's a faster/better way to do this, I'm all ears
%% 
% *Part e)* *Gradient descent solution*

alpha = 0.01;
iter = 750;

[theta_gd, cost] = gradientDescent(X_train, y_train, alpha, iter);

fprintf( ...
    "After %d iterations with alpha = %.5f:\ntheta = [%.4f; %.4f]\ncost = %.4f.\n", ...
    iter, alpha, theta_gd(1), theta_gd(2), cost(iter) ...
);

plot([1:iter], cost, '-');
ylabel('Cost');
xlabel('Iterations');
exportgraphics(gcf, 'output/ps2-4-e.png','Resolution', 200);
%% 
% *Part f) Prediciton*

% Our cost function is MSE
mse_gradient_d = computeCost(X_test, y_test, theta_gd)
%% 
% *(response)* Error shown above
%% 
% *Part g) Normal Equation*

[theta_norm] = normalEqn(X_train, y_train); % Apply norm eq.

% Our cost function is MSE
mse_norm = computeCost(X_test, y_test, theta_norm)
%% 
% *(response)* Normal equation's cost is lower than the gradient descent's cost, 
% which is expect, since Normal equation should jump right to the optimal solution.
%% 
% *Part h) Effects of learning rate*

alphas = [0.0001, 0.001, 0.003, 0.03];
iter = 250;

[discard, costs_a1] = gradientDescent(X_train, y_train, alphas(1), iter);
[discard, costs_a2] = gradientDescent(X_train, y_train, alphas(2), iter);
[discard, costs_a3] = gradientDescent(X_train, y_train, alphas(3), iter);
[discard, costs_a4] = gradientDescent(X_train, y_train, alphas(4), iter);

plot([1:iter], costs_a1, 'r-');
ylabel('Cost');
xlabel('Iterations');
exportgraphics(gcf, 'output/ps2-4-h-1.png','Resolution', 200);

plot([1:iter], costs_a2, 'g-');
ylabel('Cost');
xlabel('Iterations');
exportgraphics(gcf, 'output/ps2-4-h-2.png','Resolution', 200);

plot([1:iter], costs_a3, 'b-');
ylabel('Cost');
xlabel('Iterations');
exportgraphics(gcf, 'output/ps2-4-h-3.png','Resolution', 200);

plot([1:iter], costs_a4, 'y-');
ylabel('Cost');
xlabel('Iterations');
exportgraphics(gcf, 'output/ps2-4-h-4.png','Resolution', 200);

%% 
% *(response)* From the graphs you can kind of see how increasing alpha will 
% get you faster convergence, but going over a certain point (between 0.003 and 
% 0.03) you start to see that break down and gradient descent starts to overshoot.
%% Question 5
% *Part a) Load data*

q5_data.mat = importdata('input/hw2_data2.txt', ',', 0);

q5_data.sqft  = q5_data.mat(:, 1);
q5_data.rooms = q5_data.mat(:, 2);
q5_data.price = q5_data.mat(:, 3);

fprintf("Mean of sqft vec = %.4f, std.dev = %.4f.\n", ...
    mean(q5_data.sqft), std(q5_data.sqft) ...
);

fprintf("Mean of rooms vec = %.4f, std.dev = %.4f.\n", ...
    mean(q5_data.rooms), std(q5_data.rooms) ...
);

fprintf("Mean of price vec = %.4f, std.dev = %.4f.\n", ...
    mean(q5_data.price), std(q5_data.price) ...
);

% Lambda for normalization
norml = @(v) ( (v - mean(v)) / std(v) );

% Normalize
X_sqft  = norml(q5_data.sqft);
X_rooms = norml(q5_data.rooms);
y_price = norml(q5_data.price);

% Define X / y
X = onepad([X_sqft, X_rooms]);
y = y_price;

fprintf(...
    "The size of X is %dx%d\nThe size of y is %dx%d\n", ...
    size(X, 1), size(X, 2), size(y, 1), size(y, 2) ...
);
%% 
% *Part b) Train using gradient descent*

alpha = 0.01;
iter = 750;

[theta_housing, costs_housing] = gradientDescent(X, y, alpha, iter);

plot([1:iter], costs_housing, 'r-');
ylabel('Cost');
xlabel('Iterations');
exportgraphics(gcf, 'output/ps2-5-b.png','Resolution', 200);

fprintf( ...
    "After %d iterations with alpha = %.3f:\ntheta = [%.4f; %.4f; %.4f]\ncost = %.4f.\n", ...
    iter, alpha, theta_housing(1), theta_housing(2), theta_housing(3), costs_housing(iter) ...
);
%% 
% *(response)* See above for theta
%% 
% *Part c) Predict housing cost*

ft = [1; ...
      (1420 - mean(q5_data.sqft)) / std(q5_data.sqft); ...
      (3 - mean(q5_data.rooms)) / std(q5_data.rooms) ...
     ];

price_norml = ft' * theta_housing;

price = price_norml * std(q5_data.price) + mean(q5_data.price);

fprintf(...
    "Price of a 1420 sq ft., 3 bedroom home estimated to be $%.2f", ...
    price ...
)
