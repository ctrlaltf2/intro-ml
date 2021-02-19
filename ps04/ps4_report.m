%% *Homework Assignment 4: Regularization & Nearest Neighbors*
% *Due Sunday, February 21st, 2021 at 11:59 pm*
% 
%% *Question 1: Regularization*
%% 
% *Part a) Write an equation* |Reg_normalEqn| *that computes the closed-form 
% solution to linear regression with regularization.*

% See Reg_normalEqn.m
%% 
% *Part b) Load data*

onepad = @(x) ( [ones(length(x), 1), x] );
X_data = onepad(load('input/hw4_data1.mat', 'X_data').X_data)
y = load('input/hw4_data1.mat', 'y').y;
%% 
% *(response)* The size of the feature matrix is 1001x501.
%% 
% *Part c) Average testing error*

iter = 20;
lambdas = [0    0.001   0.003   0.005   0.007   0.009   0.012   0.017];
tst_err = zeros(iter, length(lambdas)); % For storing each err, will be mean'd
trn_err = zeros(iter, length(lambdas));
%%
% i) data splitting & iteration
M = size(y, 1);   % Total number of samples
T = floor(0.88 * M); % Number of samples to use for training (~88%)

for i = [1:iter]
    % Generate indices such that trn_idx randomly points to ~88% of the data,
    % and tst_idx to the remaining ~12% of the data
    trn_idx = randperm(M, T);          % Training data indices
    tst_idx = setdiff([1:M], trn_idx); % Testing data indices
    
    % Randomly split up the data based on the generated indices
    X_train = X_data(trn_idx, :);
    y_train = y(trn_idx, :);
    
    X_test = X_data(tst_idx, :);
    y_test = y(tst_idx, :);
    
    % ii) Use function from (1a) to train eight linear
    % regression models with varying lambdas
    for j = [1:length(lambdas)]
        lambda = lambdas(j);
        
        % Train
        theta = Reg_normalEqn(X_train, y_train, lambda);
        
        % iii) Get errors
        trn_err(i, j) = computeCost(X_train, y_train, theta);
        tst_err(i, j) = computeCost(X_test, y_test, theta);
    end
end

% iii) Compute means
tst_mean = mean(tst_err);
trn_mean = mean(trn_err);
%%
% iii) Plot
plot(lambdas, trn_mean, 'r*-');
hold on
plot(lambdas, tst_mean, 'bo-');
hold off
legend('training error', 'testing error');
ylabel('Average Error');
xlabel('λ');
exportgraphics(gcf, 'output/ps4-1-a.png','Resolution', 200);
%% 
% *(response)* The best $\lambda$ would probably be around 0.005 to 0.009. It's 
% very subtle but in the graph the difference in error between the training error 
% and testing error is at a minimum around this interval. Minimizing this difference 
% is key to having an optimal model that transfers well from training to testing.
%% Question 2 - The effect of K

load('input/hw4_data2.mat');

% Test with 5, train with the rest
X_train1 = [X1; X2; X3; X4];
y_train1 = [y1; y2; y3; y4];
X_test1  = X5;
y_test1  = y5;

% Test 4
X_train2 = [X1; X2; X3; X5];
y_train2 = [y1; y2; y3; y5];
X_test2  = X4;
y_test2  = y4;

% Test 3
X_train3 = [X1; X2; X4; X5];
y_train3 = [y1; y2; y4; y5];
X_test3  = X3;
y_test3  = y3;

% Test 2
X_train4 = [X1; X3; X4; X5];
y_train4 = [y1; y3; y4; y5];
X_test4  = X2;
y_test4  = y2;

% Test 1
X_train5 = [X2; X3; X4; X5];
y_train5 = [y2; y3; y4; y5];
X_test5  = X1;
y_test5  = y1;

Avgs = [];

for K = [1:2:15]
    % Train
    K1 = fitcknn(X_train1, y_train1, 'NumNeighbors', K);
    K2 = fitcknn(X_train2, y_train2, 'NumNeighbors', K);
    K3 = fitcknn(X_train3, y_train3, 'NumNeighbors', K);
    K4 = fitcknn(X_train4, y_train4, 'NumNeighbors', K);
    K5 = fitcknn(X_train5, y_train5, 'NumNeighbors', K);
    
    % Predict
    [Label1, discard, Cost1] = predict(K1, X_test1);
    [Label2, discard, Cost2] = predict(K2, X_test2);
    [Label3, discard, Cost3] = predict(K3, X_test3);
    [Label4, discard, Cost4] = predict(K4, X_test4);
    [Label5, discard, Cost5] = predict(K5, X_test5);
    
    % Get accuracy
    accuracy = mean([ ...
        sum(y_test1 == Label1) / length(y_test1), ...
        sum(y_test2 == Label2) / length(y_test2), ...
        sum(y_test3 == Label3) / length(y_test3), ...
        sum(y_test4 == Label4) / length(y_test4), ...
        sum(y_test5 == Label5) / length(y_test5)  ...
    ]);
    
    Avgs = [Avgs accuracy];
end
%%
plot([1:2:15], Avgs, 'gx-');
xlabel('K');
ylabel('Average Accuracy');
exportgraphics(gcf, 'output/ps4-2-a.png','Resolution', 200);
%% 
% *(response)* Having $K = 9$ would be the best option here. With cross validation 
% it provides the highest average accuracy for the given data set. This value 
% likely wouldn't necessarily transfer to different problems, since the best $K$ 
% value depends on how the data is spacially laid out.
%% Question 3 - Weighted KNN

data3 = importdata('input/hw4_data3.mat');

X_train = data3.X_train;
y_train = data3.y_train;

X_test  = data3.X_test;
y_test  = data3.y_test;
%%
% Part a) See weightedKNN.m
%%
% Part b)
sigmas = [0.01 0.1 0.5 1 3 5];
% sigmas = [0.01:0.01:7];
accuracy = [];

% Calculate accuracy for varying sigmas
for sigma = sigmas
    y_prediction = weightedKNN(X_train, y_train, X_test, sigma);
    accuracy = [ ...
                accuracy ...
                sum(y_prediction == y_test) / length(y_test) ...
               ];
end

table(sigmas', 100 * accuracy', 'VariableNames', {'Sigma', 'Accuracy %'})
%% 
% *(response)* Based on the table, the best value of sigma peaks between 0.1 
% to 1. I tried a few values manually and got ~0.62-0.8 as the best value, with 
% an accuracy of 96%.
