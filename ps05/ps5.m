%% *Homework Assignment 5: PCA & Face Recognition*
% *Due Sunday, March 14st, 2021 at 11:59 pm*
% 
%% *Question 0: Data Preprocessing*

% Cleanup & reset from past runs if applicable
if exist('input/test', 'dir')
    rmdir("input/test", 's')
end

if exist('input/train', 'dir')
    rmdir("input/train", 's')
end

mkdir('input/train');
mkdir('input/test');
%%
% For each subject folder,
subject_folder_expr = '^s\d\d?$';
for file = dir('input/all')'
    % (skip if not a subject folder)
    match = regexp(file.name, subject_folder_expr);
    if(isempty(match))
        continue
    end
    
    mkdir(['input/test/', file.name]);

    random_indices = randperm(10);
    
    % input/all/${file.name}/?.pgm -> input/train/${file.name}_k.pgm
    k = 0;
    for n = random_indices([1:8])
        src = ['input/all/', file.name, '/', num2str(n), '.pgm'];
        dst = ['input/train/', file.name, '_', num2str(k), '.pgm'];
        copyfile(src, dst);
        k = k + 1;
    end

    % input/all/${file.name}/?.pgm -> input/test/${file.name}/k.pgm (either 1 or 2)
    k = 0;
    for n = random_indices([9:10])
        src = ['input/all/', file.name, '/', num2str(n), '.pgm'];
        dst = ['input/test/', file.name, '/', num2str(k), '.pgm'];
        copyfile(src, dst);
        k = k + 1;
    end
end
%%
img = imread('input/test/s1/0.pgm');
image(img);
exportgraphics(gcf, 'output/ps5-0.png','Resolution', 200);
%% Question 1: PCA Analysis
% Part A) Training matrix

T = zeros(10304, 320);
%%
n = 1;
for file = dir('input/train/*.pgm')'
    img = imread(['input/train/', file.name]);
    T(:, n) = img(:);
    n = n + 1;
end
%%
% Generate a large overview (will be stretched a lot in one direction but
% still semi-useful/interesting
image(T)

% And show some select grayscale sections
k = 0; % At the start
imshow(T(k+[1:640], [1:320]), [])
k = floor(10304 * (1/3)); % ~1/3rd  through
imshow(T(k+[1:640], [1:320]), [])
k = floor(10304 * (2/3)); % ~2/3rds through
imshow(T(k+[1:640], [1:320]), [])
% this one looks cool so save it
exportgraphics(gcf, 'output/ps5-1-a.png','Resolution', 200);
k = 10304 - 640; % At the end
imshow(T(k+[1:640], [1:320]), [])
% Part b) Mean face

m = mean(T, 2);

mean_face_img = reshape(m, [112, 92]);
imshow(mean_face_img, [])
exportgraphics(gcf, 'output/ps5-1-b.png','Resolution', 200);
%% 
% In the image shown, you can definitely make out a face. Its very blurry, but 
% that's to be expected because it's an average of 320 face images.
% Part c) Covariance Matrix

A = T - m; % centered matrix
C = A * A';
assert(all(size(C) == [10304, 10304]));

% T not needed anymore
clear T;
%%
% Show some select grayscale sections
k = 0; % Near the start
imshow(C(k+[1:1000], k+[1:1000]), [])
k = floor(10304 * (1/3)); % ~1/3rd  through
imshow(C(k+[1:1000], k+[1:1000]), [])
k = floor(10304 * (2/3)); % ~2/3rds through
imshow(C(k+[1:1000], k+[1:1000]), [])
% this one looks cool so save it
exportgraphics(gcf, 'output/ps5-1-c.png','Resolution', 200);
k = 10304 - 1000; % At the end
imshow(C(k+[1:1000], k+[1:1000]), [])
% Part d) Eigenvalues

C_eigs = eig(A'*A);
assert(all(size(C_eigs) == [320 1]));

% A not needed
clear A;
%%
% C_eigs is asc, so reverse to make desc
C_eigs_desc = C_eigs(end:-1:1);
sum_C_eigs = sum(C_eigs);

% Since we gotta graph it anyways, just compute everything all at once
running_sums = cumsum(C_eigs_desc);
v_k = running_sums / sum_C_eigs;
%%
k = [1:length(v_k)];
plot(k([1:length(v_k)]), v_k([1:length(v_k)]), 'g-')
hold on
plot(k([1:length(v_k)]), 0.95*ones(length(v_k), 1), 'r-')
hold off
legend('v(k)', '95% threshold');
xlabel('k eigenvalues');
ylabel('% of data captured');
title('number of eigenvalues vs. % of data captured');
exportgraphics(gcf, 'output/ps5-1-d.png','Resolution', 200);
%%
% Find K that captures >= 0.95
for k = [1:length(v_k)]
    if v_k(k) >= 0.95
        break
    end
end
k
% Part e) Eigenvectors

[U, ~] = eigs(C, k);
%%
t = tiledlayout(3, 3);
for j = [1:8]
    nexttile
    imshow(reshape(U(:, j), [112, 92]), [])
end
t.TileSpacing = 'none';
t.Padding = 'none';
exportgraphics(gcf, 'output/ps5-1-e.png','Resolution', 200);
%%
fprintf(...
    "The size of U is %dx%d\n", ...
    size(U, 1), size(U, 2) ...
);

% C isn't needed anymore
clear C;
%% 
% The eigenfaces look pretty similar to the ones in the slide. The first is 
% very blurred, and as you go through each eigenface different details show up. 
% As you get towards the end of the eigenfaces (not pictured) things start to 
% get very noisy and chaotic; not much clear information is there.
%% Question 2: Feature Extraction

W_training = zeros(320, k);
y_train = zeros(320, 1);
filename_expr = 's(?<subject>\d\d?)_(?<imgnum>\d\d?).pgm$';
n = 1;
for file = dir('input/train/*.pgm')'
    groups = regexp(file.name, filename_expr, 'names');
    if(isempty(groups))
        continue;
    end
    
    % Get image data
    img = imread(['input/train/', file.name]);
    I = double(img(:));
    
    % Pull out which subject it is
    subject = str2num(groups.subject);
    assert(~isempty(subject));
    
    % Append to W_training
    W_training(n, :) = U' * (I - m);
    y_train(n) = subject;
    
    n = n + 1;
end
%%
W_testing = zeros(80, k);
y_test = zeros(80, 1);
filename_expr = 's(?<subject>\d\d?)$';
n = 1;
for file = dir('input/test/s*')'
    groups = regexp(file.name, filename_expr, 'names');
    if(isempty(groups))
        continue;
    end
    
    % Pull out which subject it is
    subject = str2num(groups.subject);
    assert(~isempty(subject));
    
    for image = dir(['input/test/', file.name, '/*.pgm'])'
        % Get image data
        img = imread(['input/test/', file.name, '/', image.name]);
        I = double(img(:));
        
        % Append to W_training then yeah
        W_testing(n, :) = U' * (I - m);
        y_test(n) = subject;
    
        n = n + 1;
    end
end
%%
% Cleanup (U is big)
clear U;
%%
fprintf(...
    "The size of W_training is %dx%d\nThe size of W_testing is %dx%d\n", ...
    size(W_training, 1), size(W_training, 2), size(W_testing, 1), size(W_testing, 2) ...
);
%% Question 3: Face Recognition
% Part a) KNN

Ks = [1 3 5 7 9]
accuracies = zeros(1, length(Ks));

% Test against multiple K's
n = 1;
for K_ = Ks
     knn = fitcknn(W_training, y_train, 'NumNeighbors', K_);
     
     [label, ~, cost] = predict(knn, W_testing);
     
     accuracies(n) = sum(label == y_test) / length(y_test);
     n = n + 1;
end
%%
table(Ks', 100*accuracies', 'VariableNames', {'K', 'Accuracy %'})
%% 
% It seems the best K is equal to 1, which would just be the nearest neighbor 
% of your test point. This seems... wrong... but I did some research on OpenCV 
% and how its face recognition models work, and it seems like they use a simple 
% nearest neighbor algorithm (i.e. KNN where K = 1) once dimensionality reduction 
% is done (which is exactly what we did for this homework!). When you think about 
% it, using K = 1 for this makes sense, since the training images will be arranged 
% in 164D space, which would make things _super_ sparse. So if a test point is 
% anywhere near any training points, it'd make sense just to go with that as our 
% guess for the label.
% Part b) SVM

labels = unique(y_train);
 
% classifier(n) = SVM classifier for label # n vs. all
linear_classifiers = cell(1, length(labels));
poly3_classifiers = cell(1, length(labels));
grbf_classifiers = cell(1, length(labels));

for label = labels'
    classes = y_train == label;
    linear_classifiers{label} = fitcsvm(W_training, classes, 'KernelFunction', 'linear');
    % Using 2nd order here. I'm not sure why, googled around and visited kinds of
    % places; MATLAB won't train a valid SVM model if I go above 2nd order.
    % There's no NaN values in my data (training or testing), so not sure
    % what's going on.
    poly3_classifiers{label}  = fitcsvm(W_training, classes, 'KernelFunction', 'polynomial', 'PolynomialOrder', 2);
    grbf_classifiers{label}   = fitcsvm(W_training, classes, 'KernelFunction', 'rbf');
end
%%
% Save to file in case MATLAB crashes (again)
save linear_classifiers linear_classifiers
save poly3_classifiers poly3_classifiers
save grbf_classifiers grbf_classifiers
%%
save y_test y_test
save W_testing W_testing
%%
scores_linear = zeros(length(labels), 80);
scores_poly3 = zeros(length(labels), 80);
scores_grbf = zeros(length(labels), 80);

% For each label, get a score for each of the testing examples that it's
% that label
for n = labels'
    % Predict (get the scores, discard the actual prediction because that's
    % irrelevant)
    [~, linear_scores] = predict(linear_classifiers{n}, W_testing);
    [~, poly3_scores] = predict(poly3_classifiers{n}, W_testing);
    [~, grbf_scores] = predict(grbf_classifiers{n}, W_testing);
    
    scores_linear(n, :) = linear_scores(:, 1)';
    scores_poly3(n, :)  = poly3_scores(:, 1)';
    scores_grbf(n, :)   = grbf_scores(:, 1)';
end
%%
% Along the row dimension, get the index of the value with the minimum score/cost
% Because the scores matrix is aligned nicely, this index is the label of
% our prediction
[~, linear_predictions] = min(scores_linear, [], 1);
[~, grbf_predictions]   = min(scores_grbf, [], 1);
[~, poly3_predictions]  = min(scores_poly3, [], 1);
%%
% Compute accuracy of predictions
linear_accuracy = sum(linear_predictions == y_test') / length(y_test);
poly3_accuracy = sum(poly3_predictions == y_test') / length(y_test);
grbf_accuracy = sum(grbf_predictions == y_test') / length(y_test);
%%
table(100*linear_accuracy, 100*poly3_accuracy, 100*grbf_accuracy, 'VariableNames', {'Linear Kernel % Accuracy', '2nd Order Polynomial % Accuracy*', 'GRBF Kernel % Accuracy'})
%% 
% Based on the table it looks like the linear kernel performed very well. The 
% others, not as much, especially the guassian kernel. The guassian kernel appears 
% to have just resulted in guessing 6 for every test point (it tried its best).
% 
% Based on these results, it looks like the SVM (with linear kernel) does a 
% bit better than the KNN with the optimal K. However, it takes a bit longer to 
% train the SVM since you have to train different SVM classifiers.
% 
% 
% 
% * I noted this above, 3rd order wasn't popping out a valid classifier model 
% object for some reason - it was 'empty' (and training 'completed' nearly instantly). 
% I couldn't find anything about it on the MATLAB forums or Googling around for 
% it. This was only happening for the polynomial model  of order >= 3; everything 
% else seemed to train just fine.
