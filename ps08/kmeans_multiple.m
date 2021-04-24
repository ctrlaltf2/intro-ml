function [ids,  means,  ssd] = kmeans_multiple(X, K, iters, R)
% Some useful constants
m = size(X, 1);  % Number of samples
n = size(X, 2);  % Feature dimensionality

% R needs incremented with the way its used as a param
% R = 0 -> one run (no restart), R = 1 -> two runs (one restart).
R = R + 1;

% squeeze(R_out(R, :, :)) is the R'th restart's computed centers
R_out = zeros(R, K, n);
ssds_out = Inf * ones(R, 1);    % ssds(n, 1) is the R'th restart's ssd
ids_out = zeros(R, m);          % store each restart cluster id as a row

% parallelize along each kmeans_single run (as these are independent)
parfor restart = [1:R]
    [ids, means, ssd] = kmeans_single(X, K, iters);
    
    R_out(restart, :, :) = means;
    ssds_out(restart, :) = ssd;
    ids_out(restart, :) = ids;
end

% Assign best restart
[ssd, best_ssd_idx] = min(ssds_out, [] , 1);
means = squeeze(R_out(best_ssd_idx, :, :));
ids = ids_out(best_ssd_idx, :);

end

