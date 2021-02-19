function y_predict = weightedKNN(X_train, y_train, X_test, sigma)
    assert(sigma ~= 0); 
    % KNN basic idea:
    % for each test thing:
    %   weight the world using the e^ (pdist(test, train) / sigma) formula
    %   for each possible label:
    %       find the X_train coordinates such that y_train == label
    %       add those (weighted) values together, store as our 'score' for it being label c.
    %   end
    %   take the biggest value out of all our scores
    %   That's our weighted KNN predicition.
    
    % Iterate row-wise through each test
    
    trn.m = size(X_test, 1); % Observations
    trn.n = size(X_train, 2); % Features
    
    y_predict = zeros(trn.m, 1);
    
    labels = unique(y_train)'; % Enumerate the unique labels we have (useful for later)

    for m = [1:trn.m]
        test_coord = X_test(m, :);
        assert( all(size(test_coord) == [1, trn.n]) , "Test matrix didn't match the same number of features as train matrix.");
        
        % Apply weighing function to all training values
        X_train_weighted = exp( (-pdist2( X_train, test_coord).^2) ./ sigma^2 );
        
        scores = zeros(length(labels), 1);
        
        % Get scores for each label
        for j = [1:length(labels)]
            label = labels(j);
            coord_lookup = (label == y_train);              % Get which weights to look at
            score = sum(X_train_weighted(coord_lookup, :)); % Sum scores of the relevant weights
            scores(j) = score;
        end
        
        [~, prediction_idx] = max(scores); % Get the prediction's index
        y_predict(m) = labels(prediction_idx); % Aaaand finally append it to our predictions list
    end
end

