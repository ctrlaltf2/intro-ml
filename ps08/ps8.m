%% *Homework Assignment 8: Bagging + Clustering*
% *Due Sunday, April 25th, 2021 at 11:59 pm*
% 
%% Parallelization Config

clear

% Set to true/false to enable/disable parallel computing inside builtin
% classifiers
ENABLE_PARALLEL = true;
% Also if you don't have parallel computing toolbox you'll also need
% to change the 'parfor' loop in kmeans_multiple.m to a 'for' loop
% (simply swap the keywords out).
%% Question 1 - Bagging with Handwriting Classification

% Part a) data reading
data = load('input/HW8_data1.mat');

X = data.X;
y = data.y;

preview_idx = randperm(size(X, 1), 25);
%%
t = tiledlayout(5, 5);
for j = [1:25]
    nexttile
    imshow(reshape(X(preview_idx(1, j), :), [20, 20]), [])
end
t.TileSpacing = 'none';
t.Padding = 'none';
exportgraphics(gcf, 'output/ps8-1-a.png','Resolution', 200);

clear preview_idx;
clear data;
%%
% Part b) Split into test/train
M = size(X, 1);      % Total number of samples
T = floor(0.90 * M); % Number of samples to use for training

trn_idx = randperm(M, T);          % Training data indices
tst_idx = setdiff([1:M], trn_idx); % Testing data indices

% Randomly split up the data based on the generated indices
X_train = X(trn_idx, :);
y_train = y(trn_idx, :);

X_test = X(tst_idx, :);
y_test = y(tst_idx, :);
%%
clear trn_idx M T tst_idx
%%
% Part c) Generate training bags
bag_idx = randperm(size(X_train, 1), 900);
X1 = X_train(bag_idx, :);
y1 = y_train(bag_idx, :);

bag_idx = randperm(size(X_train, 1), 900);
X2 = X_train(bag_idx, :);
y2 = y_train(bag_idx, :);

bag_idx = randperm(size(X_train, 1), 900);
X3 = X_train(bag_idx, :);
y3 = y_train(bag_idx, :);

bag_idx = randperm(size(X_train, 1), 900);
X4 = X_train(bag_idx, :);
y4 = y_train(bag_idx, :);

bag_idx = randperm(size(X_train, 1), 900);
X5 = X_train(bag_idx, :);
y5 = y_train(bag_idx, :);

save 'input/bags.mat' X1 X2 X3 X4 X5 y1 y2 y3 y4 y5
clear bag_idx
%%
% Part d) One v. One SVM on X1
rbf_SVM = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto');
svm_ovo_model = ...
    fitcecoc(X1, y1, ...
        'Coding', 'onevsone', ...
        'Options', statset('UseParallel', ENABLE_PARALLEL), ...
        'Learners', rbf_SVM ...
    );

[y_train_pred, ~] = predict(svm_ovo_model, X1);
[ovo_test_pred, ~] = predict(svm_ovo_model, X_test);

% Get classification error for training and testing respectively
q1d_ovo_train_accuracy = 100 * sum(y_train_pred == y1) / length(y1)
q1d_ovo_test_accuracy  = 100 * sum(ovo_test_pred == y_test) / length(y_test)
%%
% Part e) SVM One vs. All on X2
rbf_SVM = templateSVM('KernelFunction', 'rbf', 'KernelScale', 'auto');
svm_ova_model = ...
    fitcecoc(X2, y2, ...
        'Coding', 'onevsall', ...
        'Options', statset('UseParallel', ENABLE_PARALLEL), ...
        'Learners', rbf_SVM ...
    );

[y_train_pred, ~] = predict(svm_ova_model, X2);
[ova_test_pred, ~] = predict(svm_ova_model, X_test);

% Get classification error for training and testing respectively
q1e_ova_train_accuracy = 100 * sum(y_train_pred == y2) / length(y2)
q1e_ova_test_accuracy  = 100 * sum(ova_test_pred == y_test) / length(y_test)
%%
% Part f) KNN where K = 7 on X3
knnK7 = templateKNN('NumNeighbors', 7);
knn_model = ...
    fitcecoc(X3, y3, ...
        'Coding', 'onevsall', ...
        'Options', statset('UseParallel', ENABLE_PARALLEL), ...
        'Learners', knnK7 ...
    );

[y_train_pred, ~] = predict(knn_model, X3);
[knn_test_pred, ~] = predict(knn_model, X_test);

% Get classification error for training and testing respectively
q1f_knn_train_accuracy = 100 * sum(y_train_pred == y3) / length(y3)
q1f_knn_test_accuracy  = 100 * sum(knn_test_pred == y_test) / length(y_test)
%%
% Part g) Decision tree on X4
ctree_model = fitctree(X4, y4);

[y_train_pred, ~] = predict(ctree_model, X4);
[ctree_test_pred, ~] = predict(ctree_model, X_test);

% Get classification error for training and testing respectively
q1g_ctree_train_accuracy = 100 * sum(y_train_pred == y4) / length(y4)
q1g_ctree_test_accuracy  = 100 * sum(ctree_test_pred == y_test) / length(y_test)
%%
% Part h) Random forest on X5
random_forest_model = ...
    TreeBagger(100, X5, y5, ...
        'Options', statset('UseParallel', ENABLE_PARALLEL) ...
    );

[y_train_pred, ~] = predict(random_forest_model, X5);
[forest_test_pred, ~] = predict(random_forest_model, X_test);

% cell of number strings -> array of numbers
y_train_pred = cellfun(@str2num, y_train_pred);
forest_test_pred = cellfun(@str2num, forest_test_pred);

% Get classification error for training and testing respectively
q1h_cforest_train_accuracy = 100 * sum(y_train_pred == y5) / length(y5)
q1h_cforest_test_accuracy  = 100 * sum(forest_test_pred == y_test) / length(y_test)
%%
% Part i) Combine all with majority voting rule
combined_preds = [ovo_test_pred, ova_test_pred, knn_test_pred, ctree_test_pred, forest_test_pred];
majority_preds = mode(combined_preds, 2);
q1i_combined_accuracy  = 100 * sum(majority_preds == y_test) / length(y_test)
%%
% Part j) Summary table
t = ...
[
    q1d_ovo_train_accuracy,     q1d_ovo_test_accuracy; ...
    q1e_ova_train_accuracy,     q1e_ova_test_accuracy; ...
    q1f_knn_train_accuracy,     q1f_knn_test_accuracy; ...
    q1g_ctree_train_accuracy,   q1g_ctree_test_accuracy; ...
    q1h_cforest_train_accuracy, q1h_cforest_test_accuracy; ...
];

table(t(:, 1), t(:, 2), ...
    'VariableNames', {'Train Accuracy %', 'Test Accuracy %'}, ...
    'RowNames', {'SVM OvO', 'SVM OvA', 'KNN7', 'Decision Tree', 'Random Forest'} ...
)
%% 
% Over multiple runs, SVM one vs. one and one vs. all performed similarly. Out 
% of the five learners, those two tended to perform the best on the test data. 
% Over some of the runs, you could see decision tree overfitting the data a lot, 
% in some cases, getting 100% accuracy on train data, and something like 60% accuracy 
% on the test. The decision tree's accuracy also had a wider variance in accuracy 
% than the other four learners. Random forest tended to have less overfit (which 
% makes sense because its an ensemble of decision trees). In this case it _did_ 
% overfit the data, but the generalization error wasn't as bad as what you'd see 
% with a single decision tree. KNN did okay, its accuracy and generalization error 
% was about what you'd expect (similar to the PCA face recognition assignment). 
% 
% Over multiple runs, bagging tended to make the test data accuracy around at 
% or above that of SVM (the learner that tended to perform the best). I experimented 
% with seeing what the accuracy would be like if any one of the five classifiers 
% was removed, and in every case, the accuracy went down, so it's evident that 
% each learner (including the decision tree with ~65% accuracy) contributes to 
% the final answer in a positive way. I'd reason that adding more classifiers 
% to the mix would further improve the bagging test accuracy beyond ~92%.
%% Question 2 - K-means clustering & image segmentation

% Part a) See kmeans_single.m
%%
% Part b) See kmeans_multiple.m
%% 
% *Note: I added some extra values to the iteration and restarts count to get 
% a better feel for how they change the final appearance.*

% Part c) 
images = {imread('input/im1.jpg'); ...
          imread('input/im2.jpg'); ...
          imread('input/im3.png')};

% Combination configs
K = [2 3 5 7];
Iters = [1 3 7 13 20];
R = [0 5 15 25];

% Downsample config (height/width, px)
H = 100;
W = 100;

img_no = 1;
tic
for I = images'
    % Downsample, -> double
    J = I{1};
    im_resized = imresize(J, [H W]);
    im_resized = im2double(im_resized);
    
    X = reshape(im_resized, H*W, 3);
    
    for iter = Iters
        for k = K
            for r = R
                im_out = segment_kmeans(X, k, iter, r);
                filename = sprintf('output/ps8-img%d-iter%d-K%d-R%d.png', ...
                                    img_no, iter, k, r);
                imwrite(im_out, filename);
            end
        end
    end
    img_no = img_no + 1;
end
toc
%%
% Part d) Analysis / discussion

% Show increasing K for each file
figure
tK = tiledlayout(size(images, 1), size(K, 2));
tK.TileSpacing = 'none';
tK.Padding = 'none';

img_no = 1;
for img = [1:size(images, 1)]
    for k = K
        nexttile
        filename = sprintf('output/ps8-img%d-iter%d-K%d-R%d.png', ...
                            img_no, Iters(end), k, R(end));
        imshow(filename)
    end
    img_no = img_no + 1;
end
tK.XLabel.String = ['Increasing K (at max restarts and max iterations). K = ', mat2str(K), ' respectively.'];
tK.XLabel.FontSize = 8;
exportgraphics(gcf, 'output/ps8-2-d-increasing-K.png');
%% 
% From this image you can see that increasing the Ks will better approximate 
% the input image. This is because more Ks will generate more colors to use as 
% the segmented image's "pallete". K-means when applied to images will effectively 
% choose clusters in the 3d color space for a given image. These clusters will 
% correspond to the K most dominant colors in the image-- something we could call 
% the image's pallete. 

% Show increasing iters for each file
figure
tI = tiledlayout(size(images, 1), size(Iters, 2));
tI.TileSpacing = 'none';
tI.Padding = 'none';

img_no = 1;
for img = [1:size(images, 1)]
    for iter = Iters
        nexttile
        filename = sprintf('output/ps8-img%d-iter%d-K%d-R%d.png', ...
                            img_no, iter, K(end), R(1));
        imshow(filename)
    end
    img_no = img_no + 1;
end
tI.XLabel.String = ['Increasing number of iterations (at min restarts and highest K). Iters = ', mat2str(Iters), ' respectively.'];
tI.XLabel.FontSize = 8;
exportgraphics(gcf, 'output/ps8-2-d-increasing-iter.png');
%% 
% This one isn't as obvious, but it appears that in general increasing the number 
% of iterations will help the algorithm to choose better colors (centers) that 
% keep the most information in the image. An example of this is with the current 
% panda image. In the iters = 1 image (column 1), the greenery in the background 
% is very flat, not a lot of detail or information. As the algorithm is allowed 
% to progress over more iterations, you can see a better approximation of the 
% highly detailed background even though the image's pallete is still limited 
% to just 10 colors. This makes sense with the K-means RGB 3D color space intuition-- 
% the K-means algorithm is better able to navigate towards a more representative 
% center for a given cluster of pixel colors the more iterations it's given. A 
% more representative center will mean a more accurate approximation of the original 
% images in the new K-sized color pallete.

% Show increasing restarts for each file
figure
tR = tiledlayout(size(images, 1), size(R, 2));
tR.TileSpacing = 'none';
tR.Padding = 'none';

img_no = 1;
for img = [1:size(images, 1)]
    for r = R
        nexttile
        filename = sprintf('output/ps8-img%d-iter%d-K%d-R%d.png', ...
                            img_no, Iters(1), K(end), r);
        imshow(filename)
    end
    img_no = img_no + 1;
end
tR.XLabel.String = ['Increasing number of restarts (at min iterations and highest K). R = ', mat2str(R), ' respectively.'];
tR.XLabel.FontSize = 8;
exportgraphics(gcf, 'output/ps8-2-d-increasing-restarts.png');
%% 
% In this case, you can see that the effect of multiple restarts at low number 
% of iterations has a similar effect to just one run of the algorithm at high 
% iterations. The panda picture undergoes a similar transformation to the one 
% highlighted in the increasing iterations picture. 
% 
% Also something to note here, a good amount of the combinations ended up producing 
% the exact same image (down to the same sha256sum). This tended to happen with 
% low K (2 or 3), and high number of restarts and/or iterations, which makes sense 
% since there are probably less possible solutions/ssd minimums avilable with 
% K=2 or 3, and having higher iterations and/or restarts would give a higher chance 
% of finding them, because more time is spent looking.
% 
%
