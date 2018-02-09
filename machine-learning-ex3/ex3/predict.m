function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%









% =========================================================================

%a1 is X prepended by column of 1's
a1 = [ones(m, 1) X]

%z1 is X * the thetas for the first layer of the NN
z1 = a1*Theta1';

%a2 is the second layer... 
%it is the hypothesis at z1 prepended by a column of 1's
a2 = [ones(size(z1, 1), 1) sigmoid(z1)]; 

%z2 is a2 * thetas for second layer of NN
z2 = a2*Theta2';

%a3 is the output layer of the NN - the probablity that the vector of X's (image hand written number) represents a given class (that is, a specific number 0-9).
a3 = sigmoid(z2);

%find which index for a given row in a3 has the maximum probablity value and return that index (which corresponds to the number 0-9
[maxProbablity, index] = max(a3, [], 2);

p = index;



end
