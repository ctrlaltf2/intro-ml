%% *Homework Assignment 6: Bayesian Classifier*
% *Due Sunday, March 28th, 2021 at 11:59 pm*
% 
%% *Data Preprocessing*

data = readtable('input/diabetes.csv');
data = data{:, :};

X = data(:, [1:8]);
y = data(:, [9]);
%%
random_indices = randperm(768);

X_train = X(random_indices([1:540]), :);
y_train = y(random_indices([1:540]), :);

X_test  = X(random_indices([541:768]), :);
y_test  = y(random_indices([541:768]), :);
%% Question 1 - Naive-Bayes Classifier

% Part a)
X_train_0 = X_train(y_train == 0, :);
X_train_1 = X_train(y_train == 1, :);

fprintf(...
    "The size of X_train_0 is %dx%d\nThe size of X_train_1 is %dx%d\n", ...
    size(X_train_0, 1), size(X_train_0, 2), size(X_train_1, 1), size(X_train_1, 2) ...
);
%%
% Part b)
c0_means = mean(X_train_0, 1);
c0_devs  = std(X_train_0);

c1_means = mean(X_train_1, 1);
c1_devs  = std(X_train_1);

table(c0_means', c0_devs', c1_means', c1_devs', 'VariableNames', {'Class 0 Means', 'Class 0 Stdevs', 'Class 1 Means', 'Class 1 Stdevs'})
%%
% Part c)
Pw0 = 0.65;
Pw1 = 0.35;     % givens

% https://en.wikipedia.org/wiki/Normal_distribution
normal_pdf = @(mean, dev, x) (1 ./ (sqrt(2)*pi*dev) .* exp(-1/2 * ( (x - mean) ./ dev ).^2));

% Keep track of the number of correct predicitons
correct = 0;

j = 1;
for tst = X_test'
    % Part c.I)
    Pxj_given_w0 = normal_pdf(c0_means, c0_devs, tst');
    Pxj_given_w1 = normal_pdf(c1_means, c1_devs, tst');
    
    % Part c.II)
    Px_given_w0 = prod(Pxj_given_w0);
    Px_given_w1 = prod(Pxj_given_w1);
    
    % Part c.III)
    Pw0_given_x = Px_given_w0 * Pw0;
    Pw1_given_x = Px_given_w1 * Pw1;
    
    prediction = -1;
    if Pw0_given_x >= Pw1_given_x
        prediction = 0;
    else
        prediction = 1;
    end

    if(prediction == y_test(j))
        correct = correct + 1;
    end
    
    j = j + 1;
end
%%
fprintf("Accuracy for the bayesian classifier is %.2f%%", 100*(correct / length(y_test)))
%% Question 2 - Minimum Distance Classifier

% Part a)
C = cov(X_train);
%%
fprintf(...
    "The size of C is %dx%d\n", ...
    size(C, 1), size(C, 2) ...
);

image(C);
exportgraphics(gcf, 'output/ps6-2-a.png','Resolution', 200);
%%
% Part b)
mu_0 = c0_means';
mu_1 = c1_means';
%%
% Part c)
Pw0 = 0.65;
Pw1 = 0.35;     % givens

d_maha = @(x, mu, S) (sqrt( (x - mu)'*pinv(S)*(x - mu) ) );

correct = 0;

j = 1;
for tst = X_test'
    % Part c.I)
    d0 = d_maha(tst, mu_0, C);
    d1 = d_maha(tst, mu_1, C);
    
    % Part c.II)
    d0_prime = d0 - log(Pw0);
    d1_prime = d1 - log(Pw1);
    
    % Part c.II)
    prediction = -1;
    if d0_prime < d1_prime
        prediction = 0;
    else
        prediction = 1;
    end
    
    if prediction == y_test(j)
        correct = correct + 1;
    end
    
    j = j + 1;
end

fprintf("Accuracy for the Mahalanbis classifier is %.2f%%", 100*(correct / length(y_test)))
%% 
% Based on these results, it seems like the naive classifier performs better 
% than the Mahalanbis classifier.
