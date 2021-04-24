function [ids, means, ssd] = kmeans_single(X, K, iters)
%KMEANS_SINGLE Perform k-means given some samples
%   Perform k-means given some samples, X. Each row is a training example,
%   each column a feature. 

% Some useful constants
m = size(X, 1);  % Number of samples
n = size(X, 2);  % Feature dimensionality

% Final cluster membership id of each sample
ids = zeros(m, 1);

% (disabled this part) Randomly initialize means based on feature space (disabled because
% sometimes a cluster wouldn't have any points, so it couldn't get updated)
% mins = min(X, [], 1);
% maxs = max(X, [], 1);
% range = maxs - mins;    % 1xn

% Center/mean of each cluster (nth row is a coordinate of mean for cluster n)
% means = rand(K, n) .* repmat(range, [K, 1]) + mins;

% Instead initialize centers with random data (with no replacement)
mean_idx = randperm(m, K);
means = X(mean_idx, :);

for j = [1:iters]
    % Compute cluster membership:
    % Compute the euclidean distances between each sample and every center
    % Membership is the index of the minimum value in each row
    [~, ids] = min(pdist2(X, means), [], 2);
    
    % Compute cluster centers
    for l = [1:K]
        class_idx = (ids == l);
        new_mean = mean(X(class_idx, :), 1);
        means(l, :) = new_mean;
    end
end

% After finished moving around centers, compute SSD
ssd = 0;
for l = [1:K]
    class_idx = (ids == l);
    ssd = ssd + sum(sum((X(class_idx, :) - means(l, :)).^2));
end

end

