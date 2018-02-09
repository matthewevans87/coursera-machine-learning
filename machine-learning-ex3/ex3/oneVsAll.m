function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%



options = optimset('GradObj', 'on', 'MaxIter', 50);

% For each of the classes (1 to 10), find its row of thetas:
for currentClass = 1:num_labels

    %init the row of thetas to zeros. (we pass it in as a column vector of size ((n+1), 1)
    initial_theta = zeros(n + 1, 1);
    
    %this says that the "answer" for the question of "what is the probability that a given row represents the currentClass" is either 0.0 or 1.0
    %if we have 3 classes (1, 2, 3), then we need to build 3 different classifiers - one for each.
    %for the first one, the case where some input data represents a "1", we want to say that the probability it equaling "1" is 1 and the probability of it being "2" or "3" is 0.
    %so, we return a vector the same shape as y, where "when y is '1', return 1, but when its something else, return 0"...
    %The important thing is: in this case, y is NOT the _actual value_ of a given row in X.. it is instead "the probabiltiy that the row in X equals <a given current class>"
    yForCurrentClass = (y == currentClass);
    
    %the row of thetas for the current class
    currentTheta = fmincg (@(t)( lrCostFunction(t, X, yForCurrentClass, lambda) ), initial_theta, options);
    
    %all_theta is the matrix of thetas corresponding to each of the classifier functions.
    all_theta(currentClass, :) = currentTheta;
    

endfor







% =========================================================================


end
