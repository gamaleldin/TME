%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x = kron_mvprod(As, b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function evaluates the efficient multiplication of matrix (A) that
% has kronecker product structure A = kron(An, ...., A2, A1) by a vector 
% (b). This is the algorithm one from:
% Scaling multidimensional inference for structured Gaussian processes
% E Gilboa, Y Saatçi, JP Cunningham - Pattern Analysis and Machine 
% Intelligence, IEEE ?, 2015
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs:
%       - As: is a cell where each element contains the matrices
%       A1,A2,...An
%       - b: a vector.
% Outputs:
%       - x: is the result of A*b
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x = kron_mvprod(As, b)
x = b;
numDraws = size(b,2);
CTN = size(x,1);
for d = 1:length(As)
    A = As{d};
    Gd = length(A);
    X = reshape(x, Gd, CTN*numDraws/Gd);
    Z = A*X;
    Z = Z';
    x = reshape(Z, CTN,numDraws);
end
x = reshape(x, CTN*numDraws, 1);
x = reshape(x, numDraws, CTN)';
end