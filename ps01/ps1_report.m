%% *Homework Assignment 1: Intro to Machine Learning & MATLAB Programming*
% *Due Friday, January 29th, 2021 at 11:59 pm*
% 
%% *Question 1 - Regression*
% The problem I'd propose is predicting a temperature at noon *(y)* on a given 
% date in January in Pittsburgh *(x)*. Because it's the same time of day, and 
% within one season, I'd expect the plot for x vs. y to show a roughly linear 
% path. Data could be collected from <https://www.ncdc.noaa.gov/cdo-web/ NOAA's 
% CDO>. It would then be filtered to be only January data, then further filtered 
% to pull out one data point around noon for each date in January for each year. 
% This problem might end up being challenging because there might be year-to-year 
% fluctuations in data that the linear regression model might not be able to capture. 
% For example, El Niño and La Niña are an example of one weather event I know 
% of that might cause year-to-year average differences in temperature depending 
% on where you are.
%% *Question 2 - Classification*
% The problem I'd propose here, for classification is, given a piece of writing, 
% classify it into which year it was written. Here, the features would be the 
% words and their frequency of use, and the label would be their year of publication. 
% Data could be pulled from <https://www.gutenberg.org/ Project Gutenberg>. This 
% problem might turn out to be challenging for a multitude of reasons. This, of 
% course, relies on the assumption that words and frequency of use is an indicator 
% for a writing's time period. This also relies on the fact that there's enough 
% data in Project Gutenberg to train a good model. Furthermore, since Project 
% Gutenberg typically houses out-of-copyright data, this model would be limited 
% to only being able accurately classify writings that were _actually_ written 
% before 1930(ish). Anything actually written past then might as well be a wild 
% guess by the model.
%% Question 3 - Basic Operations
% *Part a.* Generate a 1,000,000 x 1 vector, $x$, of random numbers from a Gaussian 
% distribution with mean of 0 and standard deviation of 2.8

x = randn(1000000, 1);
%% 
% *Part b.* What is the min and max of the elements of $x$? What is the mean? 
% What is the standard deviation?

fprintf("min(x) = %f\n", min(x))
fprintf("max(x) = %f\n", max(x))
fprintf("std(x) = %f\n", std(x))
%% 
% *Part c.* Use the function |hist| (or any other equivalent function) to plot 
% the *normalized* histogram of your vector $x$.

histogram(x, 'Normalization', 'pdf'); % PDF: area under curve (sum) = 1
exportgraphics(gcf, 'output/ps1-3-c.png','Resolution', 200);
%% 
% *(response)* This looks like a normal distributed dataset.
%% 
% *Part d.* Add 1 to every value in vector $x$, using a loop.

p3d = x;

tic;
for i = 1:size(p3d)
    p3d(i) = p3d(i) + 1;
end
loop_elapsed = toc; % seconds

fprintf("Time for loop-based incrementer: %f seconds\n", loop_elapsed)
%% 
% *Part e.* Add 1 to every value in vector $x$, without using a loop. Display 
% the elapsed time using a method different from part d.

p3e = x;

tic;
p3e = p3e + 1;
vectorized_elapsed = toc;

disp(['Time for vectorized incrementer: ' num2str(vectorized_elapsed) ' seconds.']);
%% 
% *Part f.* Define vector $y$ whose elements are the negative numbers in $x$ 
% that are greater than $-50$. How many elements did you retrieve? Repeat this 
% step twice, compare number of retrieved elements; Is there any difference? Explain.

y = x(-50 < x & x < 0)
fprintf("y size after first filter: %d\n", length(y));

% Repeat that twice
y = y(-50 < y & y < 0);
y = y(-50 < y & y < 0);
fprintf("y size after two more filters: %d\n", length(y));
%% 
% *(response)* The number of elements retrieved is shown above. After the initial 
% filter operation out of $x$, we get some number (see above) of elements. After 
% that, for consecutive filters of $y$ for that same $-50 < y_i < 0$ predicate, 
% we get the same number of items back. Since all the elements of $y$ satisfy 
% that predicate, passing $y$ through that predicate again and again will give 
% back the same elements, and the same _number_ of elements.
%% Question 4 - Linear Algebra
% *Part a.* Define the matrix $A=\left\lbrack \begin{array}{ccc}2 & 1 & 3\\2 
% & 6 & 8\\6 & 8 & 18\end{array}\right\rbrack$in matlab. Without using loops, 
% find the minimum value in each row, maximum value in each column, largest value 
% in $A$, then compute a matrix $B$ whose elements are the square of the corresponding 
% elements in $A$.

A = [2  1  3; ...
     2  6  8; ...
     6  8 18];
 
fprintf('min value in each row: %s\n', mat2str(min(A')))
fprintf('max value in each column: %s\n', mat2str(max(A)))
fprintf('largest value in A: %d\n', max(max(A)))

B = A.^2;
fprintf('A squared, element-wise = %s\n', mat2str(B))
%% 
% *Part b.* Use Matlab to solve the system of linear equations:
% 
% $$\begin{array}{l}2x+1y+3x=1\\2x+6y+8z=3\\6x+8y+18z=5\end{array}$$

% A is already defined
b = [1; 3; 5];

% Ax=b -> x = A \ b
p4b_x = A \ b;
fprintf("[x, y, z] = [%f, %f, %f]\n", p4b_x);
%% 
% (*response)* The solution is x = 0.3, y = 0.4, z = 0.
%% 
% *Part c.* Compute (show your steps) and print the L1-norm and L2-norm for 
% each of the following two vectors:
% 
% $x_1 =\left\lbrack 0\ldotp 5\;0\;1\ldotp 5\right\rbrack$ and $x_2 =\left\lbrack 
% 1\;1\;0\right\rbrack$
% 
% *(response)*

x_1 = [0.5 0 1.5];
x_2 = [1 1 0];
%% 
% General form for the norm function is:
% 
% $${\left(\sum_{i=1} \left|x_i {\left|\right.}^p \right.\right)}^{\frac{1}{p}}$$
% 
% For L1-norm, $p$ = 1, and for L2-norm, $p$ = 2.
% 
% On $x_1$:

x_1_L1 = sum(abs(x_1).^1)^(1/1) % This simplifies, but keeping it explicit.
x_1_L2 = sum(abs(x_1).^2)^(1/2)

disp('Using norm builtin:');

fprintf("x_1_L1 (norm) = %f\n", norm(x_1, 1))
fprintf("x_1_L2 (norm) = %f\n", norm(x_1, 2))
%% 
% On $x_2$:

x_2_L1 = sum(abs(x_2).^1)^(1/1)
x_2_L2 = sum(abs(x_2).^2)^(1/2)

disp('Using norm builtin:');

fprintf("x_2_L1 (norm) = %f\n", norm(x_2, 1))
fprintf("x_2_L2 (norm) = %f\n", norm(x_2, 2))
%% Question 5.
% Write a function called normalize_rows that uses a single command to make 
% the sum of each row of the output matrix equal 1. 
% 
% *(response)* For |normalize_rows| definition, <internal:91F8F2A5 see below>

m = rand(3, 3)
mnorm = normalize_rows(m)
% row-wise sum to confirm the method works
rowsum = sum(mnorm, 2)
% Repeat
m = rand(2, 4)
mnorm = normalize_rows(m)
% row-wise sum to confirm the method works
rowsum = sum(mnorm, 2)
%% Function Definitions

function [B] = normalize_rows(A)
    % You can normalize a vector by dividing all its elements by its total sum.
    % This takes that idea and expands it to a matrix.
    
    % A row-wise sum is made, copied rightwards, then the original is divided
    % element-wise by this summed-then-copied matrix.
    B = A./repmat(sum(A, 2), 1, size(A, 2));
end
