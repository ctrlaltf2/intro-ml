%% *Homework Assignment 3: Logistic Regression*
% *Due Sunday, February 7th, 2021 at 11:59 pm*
% 
% 
%% *Question 1 - Logistic Regression*
%% 
% *Part a)* Define feature matrix $X$, and label vector $y$, output their sizes.

data = importdata('input/hw3_data1.txt', ',', 0);

% Utility lambda function to leftpad a vector with 1's
onepad = @(x) ( [ones(length(x), 1), x] );

X = onepad([data(:, 1), data(:, 2)]);
y = data(:, 3);

fprintf(...
    "The size of X is %dx%d\nThe size of y is %dx%d\n", ...
    size(X, 1), size(X, 2), size(y, 1), size(y, 2) ...
);
%% 
% *Part b)* Plot the data

% Generate indices of admitted/not admitted data
admitted_idx = find(y == 1);
not_admitted_idx = find(y == 0);

% Plot
scatter(X(admitted_idx, 2), X(admitted_idx, 3), 'k+');
hold on
scatter(X(not_admitted_idx, 2), X(not_admitted_idx, 3), 'go', 'filled');
hold off

legend('Admitted', 'Not Admitted');
xlabel('Exam 1 score');
ylabel('Exam 2 score');
exportgraphics(gcf, 'output/ps3-1-b.png','Resolution', 200);
%% 
% *Part c) Test/train dividision*

M = size(y, 1);   % Total number of samples
T = floor(0.90 * M); % Number of samples to use for training (~90%)

% Generate indices such that trn_idx randomly points to ~90% of the data,
% and tst_idx to the remaining ~10% of the data
trn_idx = randperm(M, T);          % Training data indices
tst_idx = setdiff([1:M], trn_idx); % Testing data indices (setwise diff [1 ... M] - trn_idx)

% Randomly split up the data based on the generated indices
X_train = X(trn_idx, :);
y_train = y(trn_idx, :);

X_test = X(tst_idx, :);
y_test = y(tst_idx, :);
%% 
% *Part d)* Generate sigmoid function. *See sigmoid.m for sigmoid function definition.*

Z = [-10:10];
gz = sigmoid( (Z) );

plot(Z, gz, '-');

exportgraphics(gcf, 'output/ps3-1-c.png','Resolution', 200);
%% 
% *Part e)* Cost function implementation. *See costFunction.m for costFunction 
% function definition.*

toy.x = [    ...
    1, 1, 0; ...
    1, 0, 3; ...
    1, 2, 0; ...
    1, 2, 1  ...
];

toy.y = [0; 1; 0; 1];

[J, grad] = costFunction([0 0 0]', toy.x, toy.y);

J
grad
%% 
% *(response)* J when $\theta = [0, 0, 0]^T$ is shown above.
%% 
% *Part f)* Perform minimization using MATLAB's |fminunc| builtin.

% Setup options first
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Run fminunc to get the optimal theta
[theta, cost] = ...
    fminunc(@(t) (costFunction(t, X_train, y_train)), zeros(size(X_train, 2), 1), options);

theta
cost
%% 
% *(response)* Optimal theta and cost at convergence shown above.
%% 
% *Part g)* Plotting the decision boundary

% Setup plot constants
min_x1 = min(X_train(:, 2));
min_x2 = min(X_train(:, 3));
min_x2_location = (theta(3) * min_x2 + theta(1)) / -theta(2); % From rewriting boundary eqn

% Generate indices of admitted/not admitted data
admitted_idx = find(y_train == 1);
not_admitted_idx = find(y_train == 0);

% Adapted from lecture 5 slide 6
% Plot as x_2 as a function of x_1, e.g., x_2(x_1) = (theta_1 * -x_1 - x_0) / theta_2
decision = @(x_1) ( (-theta(1) - theta(2)*x_1) / theta(3) );

% Plot training data
scatter(X_train(admitted_idx, 2), X_train(admitted_idx, 3), 'k+');
hold on
scatter(X_train(not_admitted_idx, 2), X_train(not_admitted_idx, 3), 'go', 'filled');

% Plot decision boundary
fplot(decision, [min_x1, min_x2_location]);
hold off

legend('Admitted', 'Not Admitted');
xlabel('Exam 1 score');
ylabel('Exam 2 score');
exportgraphics(gcf, 'output/ps3-1-f.png','Resolution', 200);
%% 
% *Part h)* Model performance check

hypotheses = sigmoid(X_test * theta) >= 0.5;
correct = hypotheses == y_test;

Accuracy = sum(correct) / length(y_test)
%% 
% *(response)* Accuracy was around 90%, which is pretty good.
%% 
% *Part i)* Admission probability for test1 = 45, test2 = 85

x = [1, 45, 85];
admission_probability = sigmoid(x * theta)
is_admitted = admission_probability >= 0.5
%% 
% *(response)* In this iteration, the student _was admitted with an admission 
% probability/confidence of 74.79*%.*
