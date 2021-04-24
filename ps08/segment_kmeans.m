function [im_out] = segment_kmeans(im_in, K, iters, R)
[ids, means, ssd] = kmeans_multiple(im_in, K, iters, R);

% Recolor
for id = [1:K]
    id_idx = (ids == id);

    id_count = size(im_in(id_idx, :), 1);

    im_in(id_idx, :) = repmat(means(id, :), [id_count 1]);
end

im_in = 255 * im_in;
im_in = uint8(im_in);
im_in = reshape(im_in, 100, 100, 3);

im_out = im_in;

end