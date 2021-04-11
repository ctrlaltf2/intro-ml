%% *Homework Assignment 7: Neural Networks*
% *Due Sunday, April 11th, 2021 at 11:59 pm*
% 
%% Question 0 - Read Data

data = load('input/HW7_Data.mat');
X = data.X;
y = data.y;
clear data

fprintf(...
    "The size of X is %dx%d\nThe size of y is %dx%d\n", ...
    size(X, 1), size(X, 2), size(y, 1), size(y, 2) ...
);

weights = load('input/HW7_weights_2.mat');
Theta1 = weights.Theta1;
Theta2 = weights.Theta2;
clear weights
%% Question 1 - Forward Propagation

% Part a)
[p, ~] = predict(Theta1, Theta2, X);
%% 
% *Note:* I added a 'raw' outputs return value for this alongside the scalar 
% predicted class return value. This was so the predict function could be useful 
% inside of my nnCost function.

% Part b)
accuracy_percent = 100 * (sum(p == y) / length(p))
%% Question 2 - Cost Function

disp('cost when lambda is 0:')
cost_l0 = nnCost(Theta1, Theta2, X, y, 3, 0)

disp('cost when lambda is 1:')
cost_l1 = nnCost(Theta1, Theta2, X, y, 3, 1)
%% Question 3 - Sigmoid Gradient

disp('gradient of z = [-10 0 10]^T = ')
g_prime = sigmoidGradient([-10, 0, 10]')
%% Question 4 - Backpropagation
% See sGD.m for Parts a-c and e.
% 
% Note: I modified some parameters for this neural network, namely the convergence 
% epsilon/condition and the ratio of training to testing. I made the convergence 
% epsilon smaller because my network was exiting way too early and giving 33% 
% accuracy (aka chance). I also changed the ratio of training:testing data to 
% include less training data because my network was overfitting very very easily, 
% and when you added regularization you couldn't see the overfitting/underfitting 
% effects of lower/higher regularization values.

% Part d) Learning Rate
alpha = 0.01
%%
M = size(X, 1);      % Total number of samples
T = floor(0.80 * M); % Number of samples to use for training (~85%)

trn_idx = randperm(M, T);          % Training data indices
tst_idx = setdiff([1:M], trn_idx); % Testing data indices

% Randomly split up the data based on the generated indices
X_train = X(trn_idx, :);
y_train = y(trn_idx, :);

X_test = X(tst_idx, :);
y_test = y(tst_idx, :);
%% Question 5 - Testing the Network

table1 = zeros(3, 4);
costs = zeros(3, 2);

y = 1;
for L = [0:2]
    x = 1;
    for M = [50 100]
        [T1, T2] = sGD(4, 8, 3, X_train, y_train, L, alpha, M);
        
        [tst, ~] = predict(T1, T2, X_test);
        [trn, ~] = predict(T1, T2, X_train);
        
        table1(y, x)   = 100 * (sum(trn == y_train) / length(y_train));
        table1(y, x+1) = 100 * (sum(tst == y_test)  / length(y_test));
        costs(y, x) = nnCost(T1, T2, X_train, y_train, 3, L);
        
        x = x + 2;
    end
    y = y + 1;
end
%%
table(table1(:, 1), table1(:, 2), ...
    'VariableNames', {'% Acc, Train (Epoch = 50)', '% Acc, Test (Epoch = 50)'}, ...
    'RowNames', {'λ = 0', 'λ = 1', 'λ = 2'} ...
)

table(table1(:, 3), table1(:, 4), ...
    'VariableNames', {'% Acc, Train (Epoch = 100)', '% Acc, Test (Epoch = 100)'}, ...
    'RowNames', {'λ = 0', 'λ = 1', 'λ = 2'} ...
)

table(costs(:, 1), costs(:, 3), ...
    'VariableNames', {'Cost, 50 Epochs', 'Cost, 100 Epochs'}, ...
    'RowNames', {'λ = 0', 'λ = 1', 'λ = 2'} ...
)
%% 
% These results seem to make sense. For the regularization table, comparing 
% between the relative accuracies of train vs. test you can see that the test 
% accuracy is <= the train accuracy. This makes sense, and is expected, as the 
% network really shouldn't perform better on the test data than the train data 
% (this might be either a fluke, or a signal that the test data isn't comprehensive 
% enough). The costs also seem to go down as we train more epochs, which is in 
% line with the intuition that the longer you train, the better your network gets 
% (even to the point of overfitting). As far as the effect of regularization, 
% it does seem in general that the network improves with regularization. However, 
% I'm not really seeing the effects of regularization I expected / wanted to see, 
% such as better 'transfer' ratio between train accuracy and test accuracy (indicating 
% less overfit) or a overfit then underfit relationship as we move from no regularization 
% to too much regularization. With more trials at different regularization values 
% though, I'm sure this trend would start to show up.
% 
%
